import torch
import torch.nn as nn
import torch.nn.functional as F

from parser.utils.fn import compute_log_z, viterbi, compute_label_scores
from .crf import CRF


class CRFAutoEncoder(CRF):
    def __init__(self, args):
        """

        Args:
            args ():
        """
        super(CRFAutoEncoder, self).__init__(args)
        # decoder
        self.decoder = nn.Parameter(torch.randn(args.n_words, args.n_labels))

    def __repr__(self):
        s = f"{self.__class__.__name__} (\n"
        s += f"\t Encoder:\n"
        s += f"\t\t {super().__repr__()}"
        s += f"\t Decoder:\n"
        s += f"\t\t {'weight':20}: {self.decoder.data.shape}\n"
        s += f")"
        return s

    def forward(self, words, mask):
        """

        Args:
            words ():
            mask ():

        Returns:
            emits, transitions, start, end, decoder_emits
        """
        emits, = super().forward(words, mask)
        decoder_emits = F.embedding(torch.masked_fill(words, words.ge(self.args.n_words), self.args.unk_index),
                                    torch.log_softmax(self.decoder, dim=0))
        return emits, decoder_emits

    def supervised_loss(self, args, labels, mask):
        """

        Args:
            args:
            labels (): [batch_size, seq_len]
            mask (): [batch_size, seq_len]

        Returns:

        """
        emits, decoder_emits = args
        batch_size, seq_len = mask.shape
        log_z = compute_log_z(emits, self.transitions, self.start, self.end, mask)
        # compute right labels
        crf_scores = compute_label_scores(emits + decoder_emits, self.transitions, self.start, self.end, labels, mask)
        return (log_z - crf_scores) / batch_size

    def unsupervised_loss(self, args, mask):
        """

        Args:
            args:
            mask:

        Returns:

        """
        emits, decoder_emits = args
        batch_size, seq_len = mask.shape
        log_z = compute_log_z(emits, self.transitions, self.start, self.end, mask)
        log_rec = compute_log_z(emits + decoder_emits, self.transitions, self.start, self.end, mask)
        return (log_z - log_rec) / batch_size

    def predict(self, args, mask):
        """

        Args:
            args:
            mask (): [batch_size, seq_len]

        Returns:

        """
        emits, decoder_emits = args
        return viterbi(emits + decoder_emits, self.transitions, self.start, self.end, mask)

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
