import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from parser.cmds.cmd import CMD
from parser.models import CRFMI
from parser.modules.lr_scheduler import WarmupScheduler
from parser.utils.common import pad, unk, bos
from parser.utils.data import CoNLL, Dataset, Field, SubwordField, Embedding
from parser.utils.logging import get_logger, progress_bar
from parser.utils.metric import Metric, LabelMetric

logger = get_logger(__name__)


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
                               default='data/ptb/train.conll',
                               help='path to train file')
        subparser.add_argument('--dev',
                               default='data/ptb/dev.conll',
                               help='path to dev file')
        subparser.add_argument('--test',
                               default='data/ptb/test.conll',
                               help='path to test file')

        return subparser

    def __call__(self, args):
        super(Train, self).__call__(args)

        # create dir for files
        if not os.path.exists(args.save):
            os.mkdir(args.save)

        # fields
        # word field
        self.word_field = Field('words', bos=bos, pad=pad, unk=unk)
        # subword field
        self.char_field = SubwordField("chars", bos=bos, pad=pad, unk=unk, fix_len=args.fix_len)
        # label field
        self.label_field = Field('labels')
        self.fields = CoNLL(FORM=(self.word_field, self.char_field), POS=self.label_field)

        # load dataset
        self.train_dataset = Dataset(self.fields, args.train)
        self.dev_dataset = Dataset(self.fields, args.dev)
        self.test_dataset = Dataset(self.fields, args.test)

        # build vocab
        embed = None
        if args.embed:
            embed = Embedding.load(args.embed)
        self.word_field.build(self.train_dataset, args.min_freq, embed)
        self.char_field.build(self.train_dataset)
        self.label_field.build(self.train_dataset)

        # set the data loaders
        self.train_dataset.build(args.batch_size, n_buckets=args.n_buckets, shuffle=True)
        self.dev_dataset.build(args.batch_size, n_buckets=args.n_buckets)
        self.test_dataset.build(args.batch_size, n_buckets=args.n_buckets)

        logger.info(f"Train Dateset {self.train_dataset}")
        logger.info(f"Dev Dateset {self.dev_dataset}")
        logger.info(f"Test Dateset {self.test_dataset}")

        # update args
        args.update({
            'n_words': self.word_field.vocab.n_init,
            'n_labels': len(self.label_field.vocab),
            'n_chars': len(self.char_field.vocab),
            'pad_index': self.word_field.pad_index,
            'unk_index': self.word_field.unk_index,
        })

        logger.info(f"\n{args}")

        if True or not os.path.exists(args.valid_pos):
            # get valid pos classes for words
            logger.info("Counting POS of words")
            # [n_words, n_labels]
            word_label_counter = torch.zeros(args.n_words, args.n_labels).to(args.device).float()
            for words, *_, labels in progress_bar(self.train_dataset.loader):
                mask = words.ne(args.pad_index)
                words.masked_fill_(words.ge(self.args.n_words), self.args.unk_index)
                mask, words = mask[:, 1:], words[:, 1:]
                word_label_counter += torch.einsum("nw,nl->wl",
                                                   F.one_hot(words[mask], num_classes=args.n_words).float(),
                                                   F.one_hot(labels[mask], num_classes=args.n_labels).float())

            # [n_words, n_labels], (pos appears ge one & word appears ge min freq) or word appears lt min freq
            # [n_words]
            word_freq = word_label_counter.sum(-1)
            # from dictionary according to word_threshold
            valid_pos = word_label_counter.ge(1) & word_freq.ge(args.word_threshold).unsqueeze(-1)
            valid_pos[args.unk_index] = False
            # but we need to assure all pos exist, [n_labels]
            label_appear = valid_pos.sum(0).gt(0)
            try_count = 0
            while not torch.all(label_appear) and try_count < args.n_labels:
                # just choose one not appear, may be one word contain two not appear
                # [n_words, n_not_appear]
                word_mask = torch.any(word_label_counter.t()[~label_appear].t().gt(0), dim=-1)
                temp_word_freq = word_freq.clone()
                temp_word_freq[~word_mask] = -1
                temp_word_freq[args.unk_index] = -1
                word = torch.argmax(temp_word_freq)
                valid_pos[word] = word_label_counter[word].gt(0)
                label_appear = valid_pos.sum(0).gt(0)
                try_count += 1

            self.valid_pos = valid_pos
            # set <unk> and other unknown words
            self.valid_pos[args.unk_index] = False
            unk_mask = self.valid_pos.sum(-1).eq(0)
            self.valid_pos[unk_mask] = True

            torch.save(self.valid_pos, args.valid_pos)
        else:
            self.valid_pos = torch.load(args.valid_pos, map_location=args.device)

        logger.info(f"Shape of Valid POS: {self.valid_pos.shape}")
        logger.info(f"Number of words whose POS provided: {int(self.valid_pos.sum(-1).ne(args.n_labels).sum())}")

        logger.info("Create the Model")
        model = CRFMI(args).to(args.device)
        model.load_pretrained(self.word_field.embed)
        logger.info(model)

        logger.info(f"Train the Model")
        self.train(model)

        # logger.info("Load the AutoEncoder")
        # model = CRFAutoEncoder.load(args.model).to(args.device)

        # evaluate
        # dev_loss, dev_dict_metric, dev_wo_dict_metric = self.evaluate(model, self.dev_dataset.loader)
        # logger.info(f"{'dev:':10} Loss: {dev_loss:>8.4f} W/ dict {dev_dict_metric} W/O dict {dev_wo_dict_metric}")

        # write
        # self.write_conll(model)

    def train(self, model):
        """

        Args:
            model:

        Returns:

        """
        # best
        best_dev_dict_metric, best_test_dict_metric, best_epoch, = Metric(), Metric(), 0
        min_dev_loss, min_test_loss = float("inf"), float("inf")
        # optimizer
        optimizer = Adam(model.parameters(),
                         self.args.lr,
                         (self.args.mu, self.args.nu),
                         self.args.epsilon,
                         self.args.weight_decay)
        # scheduler
        decay_steps = self.args.decay_epochs * len(self.train_dataset.loader)
        # scheduler = WarmupScheduler(optimizer, warmup_steps=100, gamma=self.args.decay ** (1 / decay_steps))
        scheduler = ExponentialLR(optimizer, self.args.decay ** (1 / decay_steps))
        for epoch in range(1, self.args.epochs + 1):
            logger.info(f"Epoch {epoch} / {self.args.epochs}:")
            start = datetime.now()
            # train
            self.train_once(model, self.train_dataset.loader, optimizer, scheduler)
            # evaluate
            dev_loss, dev_dict_metric, dev_wo_dict_metric = self.evaluate(model, self.dev_dataset.loader)
            logger.info(f"{'dev:':10} Loss: {dev_loss:>8.4f} W/ dict {dev_dict_metric} W/O dict {dev_wo_dict_metric}")
            test_loss, test_dict_metric, test_wo_dict_metric = self.evaluate(model, self.test_dataset.loader)
            logger.info(
                f"{'test:':10} Loss: {test_loss:>8.4f} W/ dict {test_dict_metric} W/O dict {test_wo_dict_metric}")

            time_spent = datetime.now() - start

            # save the model if it is the best so far
            if dev_loss < min_dev_loss:
                min_dev_loss, min_test_loss = dev_loss, test_loss
                best_dev_dict_metric, best_test_dict_metric, best_epoch = dev_dict_metric, test_dict_metric, epoch
                model.save(self.args.model)
                logger.info(f"{time_spent}s elapsed (saved)\n")
            else:
                logger.info(f"{time_spent}s elapsed\n")
        logger.info(f"Max score at epoch {best_epoch}")
        logger.info(
            f"{'dev:':10} Loss: {min_dev_loss:>8.4f} {best_dev_dict_metric}")
        logger.info(
            f"{'test:':10} Loss: {min_test_loss:>8.4f} {best_test_dict_metric}")

    def train_once(self, model, loader, optimizer, scheduler):
        """

        Args:
            model ():
            loader ():
            optimizer ():
            scheduler ():

        Returns:

        """
        model.train()
        bar = progress_bar(loader)
        for words, chars, labels in bar:
            optimizer.zero_grad()
            mask = words.ne(self.args.pad_index)
            words.masked_fill_(words.ge(self.args.n_words), self.args.unk_index)
            encoder_emits, transitions, start, end, mi_loss = model(words, chars, mask)
            # compute loss
            mask = mask[:, 1:]
            partial_pos = self.valid_pos[words][:, 1:]
            loss = model.loss(encoder_emits, transitions, start, end, partial_pos, mask) + mi_loss
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
        dict_metric = LabelMetric()
        wo_dict_metric = LabelMetric()

        total_loss = 0
        sent_count = 0
        for words, chars, labels in progress_bar(loader):
            sent_count += len(words)
            mask = words.ne(self.args.pad_index)
            words.masked_fill_(words.ge(self.args.n_words), self.args.unk_index)
            emits, transitions, start, end, mi_loss = model(words, chars, mask)
            # compute loss
            # exclude <bos>
            mask = mask[:, 1:]
            partial_pos = self.valid_pos[words][:, 1:]
            loss = model.loss(emits, transitions, start, end, partial_pos, mask) + mi_loss
            # predict
            dict_predict, wo_dict_predict = model.predict(emits, transitions, start, end, partial_pos, mask)
            dict_metric(predicts=dict_predict[mask], golds=labels[mask])
            wo_dict_metric(predicts=wo_dict_predict[mask], golds=labels[mask])
            total_loss += loss.item()
        total_loss /= sent_count
        return total_loss, dict_metric, wo_dict_metric
