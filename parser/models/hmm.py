import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from parser.modules import InvertProjector
from parser.utils.fn import viterbi, compute_label_scores, compute_log_z


class VanillaHMM(nn.Module):
    def __init__(self, args):
        super(VanillaHMM, self).__init__()
        self.args = args

        self.emits = nn.Parameter(torch.randn(args.n_words, args.n_labels))
        self.transitions = nn.Parameter(torch.randn(args.n_labels, args.n_labels))
        self.start = nn.Parameter(torch.randn(args.n_labels))
        self.end = nn.Parameter(torch.randn(args.n_labels))
        self.pretrained_embed = None

    def __repr__(self):
        s = f"{self.__class__.__name__} (\n"
        s += f"\t {'emit':20}: {self.emits.data.shape}\n"
        s += f"\t {'start':20}: {self.start.data.shape}\n"
        s += f"\t {'end':20}: {self.end.data.shape}\n"
        s += f"\t {'transitions':20}: {self.transitions.data.shape}\n"
        s += f")"
        return s

    def load_pretrained(self, embed):
        pass

    def compute_emits(self, words, mask):
        """

        Args:
            words:
            mask:

        Returns:

        """
        emits = F.embedding(words, torch.log_softmax(self.emits, dim=0))
        return emits

    def forward(self, words, mask):
        """

        Args:
            words: [batch_size, seq_len]
            mask: [batch_size, seq_len]

        Returns:

        """
        emits = self.compute_emits(words, mask)

        # compute trans
        start = torch.log_softmax(self.start, dim=-1)
        end = torch.log_softmax(self.end, dim=-1)
        transitions = torch.log_softmax(self.transitions, dim=-1)

        return [emits, transitions, start, end]

    def supervised_loss(self, args, labels, mask):
        """

        Args:
            args:
            labels:
            mask:

        Returns:

        """
        emits, transitions, start, end = args
        batch_size, *_ = emits.shape
        return -compute_label_scores(emits, transitions, start, end, labels, mask) / batch_size

    def unsupervised_loss(self, args, mask):
        """

        Args:
            args:
            mask:

        Returns:

        """
        emits, transitions, start, end = args
        batch_size, *_ = emits.shape
        return -compute_log_z(emits, transitions, start, end, mask) / batch_size

    def predict(self, args, mask):
        """

        Args:
            args:
            mask:

        Returns:

        """
        emits, transitions, start, end = args
        return viterbi(emits, transitions, start, end, mask)

    def save(self, path):
        """

        Args:
            path ():

        Returns:

        """
        state_dict = self.state_dict()
        pretrained_embed = None
        if self.pretrained_embed is not None:
            pretrained_embed = state_dict.pop("pretrained_embed.weight")
        state = {
            "state": state_dict,
            "pretrained_embed": pretrained_embed,
            "args": self.args,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        """

        Args:
            path ():

        Returns:

        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state["args"]).to(device)
        model.load_pretrained(state["pretrained_embed"])
        model.load_state_dict(state["state"], strict=False)
        return model


class NeuralHMM(VanillaHMM):
    def __init__(self, args):
        super(NeuralHMM, self).__init__(args)
        self.pretrained_embed = None
        # emit computer
        self.emit_linear = nn.Linear(args.n_embed, args.n_labels)

    def __repr__(self):
        s = f"{self.__class__.__name__} (\n"
        if self.pretrained_embed is not None:
            s += f"\t {'pretrained embed':20}: {self.pretrained_embed.data.shape}\n"
        s += f"\t {'emit linear':20}: {self.emit_linear}\n"
        s += f"\t {'start':20}: {self.start.data.shape}\n"
        s += f"\t {'end':20}: {self.end.data.shape}\n"
        s += f"\t {'transitions':20}: {self.transitions.data.shape}\n"
        s += f")"
        return s

    def load_pretrained(self, embed):
        """

        Args:
            embed ():

        Returns:

        """
        if embed is not None:
            self.pretrained_embed = embed[:self.args.n_words].to(self.args.device)
            self.pretrained_embed.requires_grad = False

    def compute_emits(self, words, mask):
        """

        Args:
            words:
            mask:

        Returns:

        """
        # get embed
        words = torch.masked_fill(words, words.ge(self.args.n_words), self.args.unk_index)
        # compute emit
        emits = F.embedding(words, torch.log_softmax(self.emit_linear(self.pretrained_embed), dim=0))

        return emits

    def save(self, path):
        """

        Args:
            path ():

        Returns:

        """
        state_dict = self.state_dict()
        state = {
            "state": state_dict,
            "pretrained_embed": self.pretrained_embed,
            "args": self.args,
        }
        torch.save(state, path)


class GaussianHMM(VanillaHMM):
    def __init__(self, args):
        super(GaussianHMM, self).__init__(args)
        self.pretrained_embed = None

        #
        self.n_embed = args.n_embed
        self.n_labels = args.n_labels

        self.means = nn.Parameter(torch.Tensor(self.n_labels, self.n_embed))
        self.var = nn.Parameter(torch.ones(self.n_embed), requires_grad=False)
        self.factor = -self.n_embed / 2.0 * (math.log(2) + math.log(np.pi))
        self.log_density = self.factor - 0.5 * self.n_embed

        # invertible neural network
        self.inverter = InvertProjector(args.n_inverter_layer, args.n_embed, args.n_inverter_hidden)

    def __repr__(self):
        s = f"{self.__class__.__name__} (\n"
        s += f"\t {'pretrained embed':20}: {self.pretrained_embed}\n"
        s += f"\t {'start':20}: {self.start.data.shape}\n"
        s += f"\t {'end':20}: {self.end.data.shape}\n"
        s += f"\t {'transitions':20}: {self.transitions.data.shape}\n"
        s += f")"
        return s

    def load_pretrained(self, embed):
        """

        Args:
            embed ():

        Returns:

        """
        if embed is not None:
            self.pretrained_embed = nn.Embedding.from_pretrained(embed).to(self.args.device)

    def init_params(self):
        """

        Args:

        Returns:

        """
        with torch.no_grad():
            var, means = torch.var_mean(self.pretrained_embed.weight[:self.args.n_words], dim=0, unbiased=False)
        noisy_means = torch.zeros_like(self.means.data)
        noisy_means.normal_().mul_(0.04)
        noisy_means = noisy_means + means.unsqueeze(0)
        self.means.data.copy_(noisy_means)
        self.var.data.copy_(var)
        # update log_density
        self.log_density = self.factor - 0.5 * self.var.log().sum()

    def compute_emits(self, words, mask):
        """

        Args:
            words:
            mask:

        Returns:

        """
        # compute emits
        batch_size, seq_len = words.shape
        # get embed
        word_embed = self.pretrained_embed(words)

        ep_size = torch.Size([batch_size, seq_len, self.n_labels, self.args.n_embed])
        word_embed = word_embed.unsqueeze(dim=2).expand(ep_size)
        means = self.means.expand(ep_size)
        var = self.var.expand(ep_size)

        emits = self.log_density - 0.5 * ((means - word_embed) ** 2 / var).sum(dim=-1)

        return emits
