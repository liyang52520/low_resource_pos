import argparse
import os
from collections import Counter
from datetime import datetime

import numpy as np
import torch

from modules import read_conll, to_input_tensor, InvertGHMM, data_iter, generate_seed, sents_to_vec
from modules.config import Config
from modules.embedding import Embedding
from modules.log import get_logger, init_logger, progress_bar
from modules.metric import UnsupervisedPOSMetric
from modules.vocab import Vocab
from nlp_commons.fn import pad_fn

logger = get_logger(__name__)


def init_config():
    parser = argparse.ArgumentParser(description='POS tagging')

    # train and test data
    parser.add_argument('--word_vec', type=str,
                        help='the word vector file (cPickle saved file)')
    parser.add_argument('--train_file', type=str, help='train data')
    parser.add_argument('--evaluate_file', type=str, help='evaluate data')
    parser.add_argument('--test_file', default='', type=str, help='test data')

    # optimization parameters
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')

    # model config
    parser.add_argument('--model', choices=['gaussian', 'nice'], default='gaussian')

    # pretrained model options
    parser.add_argument('--load_nice', default='', type=str,
                        help='load pretrained projection model, ignored by default')
    parser.add_argument('--load_gaussian', default='', type=str,
                        help='load pretrained Gaussian model, ignored by default')

    # Others
    parser.add_argument('--tag_from', default='', type=str,
                        help='load pretrained model and perform tagging')
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--set_seed', action='store_true', default=False, help='if set seed')

    # these are for slurm purpose to save model
    # they can also be used to run multiple random restarts with various settings,
    # to save models that can be identified with ids
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')
    parser.add_argument('--save',
                        default='save/master',
                        help='path to saved files')

    parser.add_argument('--ud',
                        action="store_true",
                        help='run UD data')

    args = parser.parse_args()
    args = Config("config.ini").update(vars(args))

    args.cuda = torch.cuda.is_available()

    # set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().isoformat().split(".")[0]
    timestamp = str.replace(timestamp, ':', '-')
    init_logger(logger, f"{args.save}/train-{args.model}-{timestamp}.log")

    save = args.save

    if not os.path.exists(save):
        os.makedirs(save)

    id_ = "%s_%dlayers" % (args.model, args.couple_layers)
    save_path = os.path.join(save, id_ + '.pt')
    args.save_path = save_path
    args.predict_path = os.path.join(save, "dev_predict.conll")
    # if args.model == 'nice':
    #     args.load_gaussian = os.path.join(args.save, args.load_gaussian)
    if args.tag_from != '':
        if args.model == 'nice':
            args.load_nice = args.tag_from
        else:
            args.load_gaussian = args.tag_from
        args.tag_path = "pos_%s_%slayers_tagging%d_%d.txt" % \
                        (args.model, args.couple_layers, args.jobid, args.taskid)

    return args


class CMD(object):
    def __init__(self, args):
        self.args = args
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        # load word vec
        logger.info("Loading pretrained word embeddings")
        if args.ud:
            word_vec = Embedding.load(args.word_vec, unk=self.unk_token)
        else:
            import pickle
            word_vec = pickle.load(open(args.word_vec, 'rb'))

        # load data
        train_text, null_index = read_conll(args.train_file, prc_num=False, ud=args.ud)
        evaluate_text, null_index = read_conll(args.evaluate_file, prc_num=False, ud=args.ud)
        test_text, null_index = read_conll(args.test_file, prc_num=False, ud=args.ud)

        self.train_data, self.train_words = sents_to_vec(word_vec, train_text)
        self.evaluate_data, self.evaluate_words = sents_to_vec(word_vec, evaluate_text)
        self.test_data, self.test_words = sents_to_vec(word_vec, test_text)

        logger.info(f"Train Dataset {len(self.train_data)}")
        logger.info(f"Dev Dataset {len(self.evaluate_data)}")
        logger.info(f"Test Dataset {len(self.test_data)}")

        self.train_tags = [sent["tag"] for sent in train_text]
        self.test_tags = [sent["tag"] for sent in test_text]
        self.evaluate_tags = [sent["tag"] for sent in evaluate_text]

        logger.info('complete reading data')

        logger.info('Training sentences: %d' % len(self.train_data))
        logger.info(f'Evaluate sentences: {len(self.evaluate_data)}')
        logger.info('Testing sentences: %d' % len(self.test_data))

        args.n_embed = len(self.train_data[0][0])
        self.pad = np.zeros(args.n_embed)
        device = torch.device("cuda" if args.cuda else "cpu")

        logger.info("Building word vocab")
        all_words = []
        for words in progress_bar(self.train_words):
            all_words.extend(words)
        word_counter = Counter(all_words)
        word_vocab = Vocab(counter=word_counter, min_freq=1, specials=[self.pad_token, self.unk_token], unk_index=1)
        pad_index = word_vocab[self.pad_token]
        unk_index = word_vocab[self.unk_token]
        self.word_vocab = word_vocab

        logger.info("Transfer words to idx")
        self.train_words = [word_vocab.transfer(words) for words in self.train_words]
        self.evaluate_words = [word_vocab.transfer(words) for words in self.evaluate_words]
        self.test_words = [word_vocab.transfer(words) for words in self.test_words]

        logger.info("Building tag vocab")
        all_tags = []
        for tags in progress_bar(self.train_tags):
            all_tags.extend(tags)
        tag_counter = Counter(all_tags)
        tag_vocab = Vocab(counter=tag_counter, min_freq=1)
        self.tag_vocab = tag_vocab

        logger.info("Transfer tags to idx")
        self.train_tags = [tag_vocab.transfer(tags) for tags in self.train_tags]
        self.evaluate_tags = [tag_vocab.transfer(tags) for tags in self.evaluate_tags]
        self.test_tags = [tag_vocab.transfer(tags) for tags in self.test_tags]

        self.evaluate_loader = list(data_iter(list(zip(self.evaluate_data, self.evaluate_words, self.evaluate_tags)),
                                              batch_size=self.args.batch_size, is_test=True, shuffle=False))
        self.test_loader = list(data_iter(list(zip(self.test_data, self.test_words, self.test_tags)),
                                          batch_size=self.args.batch_size, is_test=True, shuffle=False))

        # update some args
        args.update({
            "n_words": len(word_vocab),
            "n_labels": len(tag_vocab),
            "pad_index": pad_index,
            "unk_index": unk_index,
            "n_embed": args.n_embed,
            "device": device,
        })

        logger.info(f"\n{args}\n")

        if self.args.load_nice != "":
            self.args.load_nice = os.path.join(self.args.save, args.load_nice)
            model = InvertGHMM.load(self.args.load_nice)
        elif self.args.load_gaussian != "":
            self.args.load_gaussian = os.path.join(self.args.save, args.load_gaussian)
            model = InvertGHMM.load(self.args.load_gaussian)
        else:
            # create and init model
            model = InvertGHMM(args).to(device)
            init_seed = to_input_tensor(generate_seed(self.train_data, args.batch_size), self.pad, device=device)
            model.init_params(init_seed)

        # init test
        logger.info("Evaluate the model")
        with torch.no_grad():
            model.eval()
        dev_loss, dev_metric = self.evaluate(model, self.evaluate_loader)
        logger.info(f'Result on Dev: {dev_metric}, Loss {dev_loss:>8.4f}')
        test_loss, test_metric = self.evaluate(model, self.test_loader)
        test_metric.set_match(*dev_metric.match)
        logger.info(f'Result on Test: {test_metric}, Loss {test_loss:>8.4f}\n')

        # train the model
        logger.info(f"Train the Model")
        self.train(model)

        # write the model
        # dir_path = "de_all"
        # train_loader = list(data_iter(list(zip(self.train_data, self.train_words, self.train_tags)),
        #                               batch_size=self.args.batch_size, is_test=True, shuffle=False))
        # self.write(model, train_loader, os.path.join("gaussian_de_all", f"total.{args.seed}.conll"))
        # self.write(model, self.evaluate_loader, os.path.join(dir_path, f"total.{args.seed}.conll"))
        # self.write(model, self.test_loader, os.path.join(dir_path, f"total.{args.seed}.conll"))

    def train(self, model):
        """

        Args:
            model:

        Returns:

        """
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), self.args.lr)

        # train
        best_dev_metric = UnsupervisedPOSMetric(n_clusters=self.args.n_labels, device=self.args.device)
        best_test_metric = UnsupervisedPOSMetric(n_clusters=self.args.n_labels, device=self.args.device)
        best_dev_loss, best_test_loss = float("inf"), float("inf")
        best_epoch = 0
        for epoch in range(1, self.args.epochs + 1):
            logger.info(f"Epoch {epoch} / {self.args.epochs}: ")
            start = datetime.now()
            # build loader
            data_loader = list(data_iter(list(zip(self.train_data, self.train_words, self.train_tags)),
                                         batch_size=self.args.batch_size, is_test=True, shuffle=True))
            self.train_once(model, optimizer, data_loader)

            # evaluate
            with torch.no_grad():
                model.eval()
                dev_loss, dev_metric = self.evaluate(model, self.evaluate_loader)
                logger.info(f'Dev: {dev_metric}, Loss {dev_loss:>8.4f}')
                # test_loss, test_metric = self.evaluate(model, self.test_loader)
                # test_metric.set_match(*dev_metric.match)
                # logger.info(f'Test: {test_metric}, Loss {test_loss:>8.4f}')

            time_spent = datetime.now() - start
            # save model
            if dev_loss < best_dev_loss:
                # best_dev_metric, best_test_metric = dev_metric, test_metric
                # best_dev_loss, best_test_loss = dev_loss, test_loss
                best_dev_metric = dev_metric
                best_dev_loss = dev_loss
                best_epoch = epoch
                model.save(self.args.save_path)
                logger.info(f"{time_spent}s elapsed (saved)\n")
            else:
                logger.info(f"{time_spent}s elapsed\n")

        logger.info(f"Max score at epoch {best_epoch}")
        logger.info(f"{'dev:':10} Loss: {best_dev_loss:>8.4f} {best_dev_metric}")
        # logger.info(f"{'test:':10} Loss: {best_test_loss:>8.4f} {best_test_metric}\n")

    def train_once(self, model, optimizer, data_loader):
        """

        Args:
            model:
            optimizer:
            data_loader:

        Returns:

        """
        model.train()

        bar = progress_bar(data_loader)
        for sents, words, labels in bar:
            optimizer.zero_grad()
            # [seq_len, batch_size]
            words = pad_fn(words, padding_value=self.args.pad_index).t().long().to(self.args.device)
            # get embed
            pretrained_embed, masks = to_input_tensor(sents, self.pad, device=self.args.device)
            # loss
            loss = model.unsupervised_loss(words, pretrained_embed, masks)
            loss.backward()
            optimizer.step()
            bar.set_postfix_str(f'loss: {loss.item():.4f}')

    def evaluate(self, model, data_loader):
        """

        Args:
            model:
            data_loader:

        Returns:

        """
        metric = UnsupervisedPOSMetric(n_clusters=self.args.n_labels, device=self.args.device)

        total_loss = 0.0
        sent_count = 0.0
        for sents, words, labels in progress_bar(data_loader):
            sent_count += len(sents)
            # [seq_len, batch_size]
            words = pad_fn(words, padding_value=self.args.pad_index).t().long().to(self.args.device)
            # [batch_size, seq_len]
            labels = pad_fn(labels).long()
            # get embed
            pretrain_embed, mask = to_input_tensor(sents, self.pad, device=self.args.device)
            # loss
            total_loss += model.unsupervised_loss(words, pretrain_embed, mask).item()
            # [batch_size, seq_length]
            predicts = model.predict(pretrain_embed, mask)
            mask = mask.t().bool()
            metric(predicts=predicts[mask], golds=labels[mask])

        total_loss /= sent_count

        return total_loss, metric

    def write(self, model, data_loader, path):
        """

        Args:
            model:
            data_loader:
            path:

        Returns:

        """
        model.eval()
        out_file = open(path, "w", encoding="utf8")
        bar = progress_bar(data_loader)
        for sents, words, labels in bar:
            # [seq_len, batch_size]
            words = pad_fn(words, padding_value=self.args.pad_index).t().long().to(self.args.device)
            # [batch_size, seq_len]
            labels = pad_fn(labels).long()
            # get embed
            pretrained_embed, masks = to_input_tensor(sents, self.pad, device=self.args.device)
            # loss
            predicts = model.predict(pretrained_embed, masks)
            masks = masks.t().bool()
            words = words.t()
            for word, predict, label, mask in zip(words, predicts, labels, masks):
                word, label, predict = word[mask].tolist(), label[mask].tolist(), predict[mask].tolist()
                # write file
                for idx in range(len(word)):
                    print(
                        f"{idx}\t{self.word_vocab[word[idx]]}\t-\t{predict[idx]}\t{self.tag_vocab[label[idx]]}\t-\t-\t-\t-\t-",
                        file=out_file)
                print(file=out_file)
        out_file.close()


if __name__ == '__main__':
    parse_args = init_config()
    cmd = CMD(parse_args)
