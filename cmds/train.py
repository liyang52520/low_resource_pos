import os
from datetime import datetime

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from cmds.cmd import CMD
from parser.models import CRFAutoEncoder, CRF, VanillaHMM, NeuralHMM, GaussianHMM
from parser.utils.common import pad, unk
from parser.utils.data import CoNLL, Dataset, Field, Embedding
from parser.utils.logging import get_logger, progress_bar
from parser.utils.metric import Metric, LabelMetric

logger = get_logger(__name__)

model_classes = {
    "crf": CRF,
    "crf_ae": CRFAutoEncoder,
    "vanilla_hmm": VanillaHMM,
    "neural_hmm": NeuralHMM,
    "gaussian_hmm": GaussianHMM,
}


class Train(CMD):
    def create_subparser(self, parser, mode):
        """
        create a subparser and add arguments in this subparser
        Args:
            parser: parent parser
            mode (str): mode of this subparser

        Returns:

        """
        subparser = parser.add_parser(mode, help='Train model.')
        subparser.add_argument('--train',
                               default='data/corpus/ptb/train.conll',
                               help='path to train file')
        subparser.add_argument('--train-labeled',
                               default='data/corpus/ptb/train.100.conll',
                               help='path to train file')
        subparser.add_argument('--dev',
                               default='data/corpus/ptb/dev.conll',
                               help='path to dev file')
        subparser.add_argument('--test',
                               default='data/corpus/ptb/test.conll',
                               help='path to test file')
        subparser.add_argument("--mixed_step",
                               type=int,
                               default="1",
                               help="ration between unsupervised data and supervised data")

        return subparser

    def __call__(self, args):
        super(Train, self).__call__(args)

        # create dir for files
        if not os.path.exists(args.save):
            os.mkdir(args.save)

        # fields
        # word field
        self.word_field = Field('words', pad=pad, unk=unk)
        # label field
        self.label_field = Field('labels')
        self.fields = CoNLL(FORM=self.word_field, POS=self.label_field)

        # load dataset
        self.train_dataset = Dataset(self.fields, args.train)
        self.train_labeled_dataset = Dataset(self.fields, args.train_labeled)
        self.dev_dataset = Dataset(self.fields, args.dev)
        self.test_dataset = Dataset(self.fields, args.test)

        # build vocab
        embed = None
        if args.embed is not None:
            embed = Embedding.load(args.embed)
        self.word_field.build(self.train_dataset, args.min_freq, embed)
        self.label_field.build(self.train_dataset)

        # set the data loaders
        self.train_dataset.build(args.batch_size, n_buckets=args.n_buckets, shuffle=True)
        self.train_labeled_dataset.build(args.batch_size, n_buckets=args.n_buckets, shuffle=True)
        self.dev_dataset.build(args.batch_size, n_buckets=args.n_buckets)
        self.test_dataset.build(args.batch_size, n_buckets=args.n_buckets)

        logger.info(f"Train Dateset {self.train_dataset}")
        logger.info(f"Train Labeled Dateset {self.train_labeled_dataset}")
        logger.info(f"Dev Dateset {self.dev_dataset}")
        logger.info(f"Test Dateset {self.test_dataset}")

        # update args
        args.update({
            'n_words': self.word_field.vocab.n_init,
            'n_labels': len(self.label_field.vocab),
            'pad_index': self.word_field.pad_index,
            'unk_index': self.word_field.unk_index,
        })

        logger.info(f"\n{args}")

        logger.info("Create the Model")
        model_class = model_classes[args.model]

        model = model_class(args).to(args.device)
        model.load_pretrained(self.word_field.embed)

        # init model
        if args.model == "gaussian_hmm":
            model.init_params()

        logger.info(model)

        logger.info(f"Train the Model with Labeled Data")
        self.train(model)

        logger.info(f"Load the Best Model")
        model = model_class.load(args.model_path)

        if args.model != "crf":
            logger.info(f"\n\nTrain the Model with Mixed Data")
            self.train(model, unsupervised=True)

        # invert has two steps for training
        if args.model == "gaussian_hmm" and self.args.invert:
            model = model_class.load(args.model_path)
            model.args.invert = True
            logger.info(f"Train the Invert with Labeled Data")
            self.train(model)
            logger.info(f"Load the Best Model")
            model = model_class.load(args.model_path)
            logger.info(f"Train the Invert with Mixed Data")
            self.train(model, unsupervised=True)

    def train(self, model, unsupervised=False):
        """

        Args:
            model:
            unsupervised

        Returns:

        """
        # best
        best_dev_metric, best_test_metric, best_epoch, = Metric(), Metric(), 0
        min_dev_loss, min_test_loss = float("inf"), float("inf")
        # optimizer
        optimizer = Adam(model.parameters(),
                         self.args.lr,
                         (self.args.mu, self.args.nu),
                         self.args.epsilon,
                         self.args.weight_decay)

        # scheduler
        decay_steps = self.args.decay_epochs * len(self.train_dataset.loader)
        scheduler = ExponentialLR(optimizer, self.args.decay ** (1 / decay_steps))
        for epoch in range(1, self.args.epochs + 1):
            logger.info(f"Epoch {epoch} / {self.args.epochs}:")
            start = datetime.now()
            # train
            if not unsupervised:
                self._train_supervised(model,
                                       self.train_labeled_dataset.loader,
                                       optimizer,
                                       scheduler)
            else:
                self._train_unsupervised(model,
                                         self.train_dataset.loader,
                                         self.train_labeled_dataset.loader,
                                         optimizer,
                                         scheduler)
            # evaluate
            dev_loss, dev_metric = self.evaluate(model, self.dev_dataset.loader)
            logger.info(f"{'dev:':10} Loss: {dev_loss:>8.4f} {dev_metric}")
            test_loss, test_metric = self.evaluate(model, self.test_dataset.loader)
            logger.info(f"{'test:':10} Loss: {test_loss:>8.4f} {test_metric}")

            time_spent = datetime.now() - start

            # save the model if it is the best so far
            if dev_metric > best_dev_metric:
                min_dev_loss, min_test_loss = dev_loss, test_loss
                best_dev_metric, best_test_metric, best_epoch = dev_metric, test_metric, epoch
                model.save(self.args.model_path)
                logger.info(f"{time_spent}s elapsed (saved)\n")
            else:
                logger.info(f"{time_spent}s elapsed\n")

        logger.info(f"Max score at epoch {best_epoch}")
        logger.info(f"{'dev:':10} Loss: {min_dev_loss:>8.4f} {best_dev_metric}")
        logger.info(f"{'test:':10} Loss: {min_test_loss:>8.4f} {best_test_metric}")

    def _train_supervised(self, model, labeled_loader, optimizer, scheduler, show=True):
        """

        Args:
            model ():
            labeled_loader ():
            optimizer ():
            scheduler ():

        Returns:

        """
        model.train()
        bar = progress_bar(labeled_loader) if show else labeled_loader
        for words, labels in bar:
            optimizer.zero_grad()
            mask = words.ne(self.args.pad_index)
            # compute loss
            args = model(words, mask)
            loss = model.supervised_loss(args, labels, mask)
            loss.backward()
            #
            nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()
            scheduler.step()
            if show:
                bar.set_postfix_str(f" lr: {scheduler.get_last_lr()[0]:.4e} , loss: {loss.item():.4f}")

    def _train_unsupervised(self, model, unlabeled_loader, labeled_loader, optimizer, scheduler):
        """

        Args:
            model ():
            unlabeled_loader ():
            labeled_loader ():
            optimizer ():
            scheduler ():

        Returns:

        """
        model.train()
        bar = progress_bar(unlabeled_loader)
        count = 0
        for words, labels in bar:
            count += 1
            if count % self.args.mixed_step == 0:
                self._train_supervised(model, labeled_loader, optimizer, scheduler, False)
            optimizer.zero_grad()
            mask = words.ne(self.args.pad_index)
            # compute loss
            args = model(words, mask)
            loss = model.unsupervised_loss(args, mask)
            loss.backward()
            #
            nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()
            scheduler.step()
            bar.set_postfix_str(f" lr: {scheduler.get_last_lr()[0]:.4e} , loss: {loss.item():.4f}")

    def evaluate(self, model, loader):
        """

        Args:
            model:
            loader:

        Returns:

        """
        model.eval()
        metric = LabelMetric()

        total_loss = 0
        sent_count = 0
        for words, labels in progress_bar(loader):
            sent_count += len(words)
            mask = words.ne(self.args.pad_index)
            args = model(words, mask)
            loss = model.supervised_loss(args, labels, mask)
            # predict
            predict = model.predict(args, mask)
            metric(predicts=predict[mask], golds=labels[mask])
            total_loss += loss.item()
        total_loss /= sent_count
        return total_loss, metric
