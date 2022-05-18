import torch
import torch.nn as nn
import torch.nn.functional as F

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
