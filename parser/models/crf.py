import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parser.modules import BiLSTM
from parser.modules.dropout import SharedDropout
from parser.utils.fn import compute_log_z, viterbi, compute_label_scores


class CRF(nn.Module):
    def __init__(self, args):
        """

        Args:
            args ():
        """
        super(CRF, self).__init__()
        self.args = args

        self.word_embed = nn.Embedding(args.n_words, args.n_embed)
        self.pretrained_embed = None
        self.embed_dropout = nn.Dropout(args.dropout)

        self.lstm = BiLSTM(input_size=args.n_embed,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layer,
                           dropout=args.dropout)
        self.lstm_dropout = SharedDropout(args.dropout)

        self.emit_linear = nn.Linear(args.n_lstm_hidden * 2, args.n_labels)

        self.start = nn.Parameter(torch.randn(args.n_labels))
        self.end = nn.Parameter(torch.randn(args.n_labels))
        self.transitions = nn.Parameter(torch.randn(args.n_labels, args.n_labels))

    def __repr__(self):
        s = f"{self.__class__.__name__} (\n"
        s += f"\t {'word embed':20}: {self.word_embed}\n"
        s += f"\t {'pretrained embed':20}: {self.pretrained_embed}\n"
        s += f"\t {'lstm':20}: {self.lstm}\n"
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
            self.pretrained_embed = nn.Embedding.from_pretrained(embed).to(self.args.device)
            nn.init.zeros_(self.word_embed.weight)

    def forward(self, words, mask):
        """

        Args:
            words ():
            mask ():

        Returns:
            emits, transitions, start, end
        """
        batch_size, seq_len = words.shape
        # get embed
        ext_words = words
        if self.pretrained_embed is not None:
            ext_words = torch.masked_fill(words, words.ge(self.args.n_words), self.args.unk_index)
        word_embed = self.word_embed(ext_words)
        if self.pretrained_embed is not None:
            word_embed = word_embed + self.pretrained_embed(words)
        word_embed = self.embed_dropout(word_embed)

        # LSTM
        x = pack_padded_sequence(word_embed, mask.sum(1).cpu(), True, False)
        x, _ = self.lstm(x)
        # [batch_size, seq_len, n_lstm_hidden * 2]
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # [batch_size, seq_len, n_labels]
        emits = self.emit_linear(x)
        # start and end
        return emits,

    def supervised_loss(self, args, labels, mask):
        """

        Args:
            args:
            labels (): [batch_size, seq_len]
            mask (): [batch_size, seq_len]

        Returns:

        """
        emits, = args
        batch_size, seq_len = mask.shape
        log_z = compute_log_z(emits, self.transitions, self.start, self.end, mask)
        # compute right labels
        crf_scores = compute_label_scores(emits, self.transitions, self.start, self.end, labels, mask)
        return (log_z - crf_scores) / batch_size

    def predict(self, args, mask):
        """

        Args:
            args:
            mask (): [batch_size, seq_len]

        Returns:

        """
        emits, = args
        return viterbi(emits, self.transitions, self.start, self.end, mask)

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
