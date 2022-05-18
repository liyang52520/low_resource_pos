import math

import numpy as np
from torch.nn import Parameter

from .projection import *
from .utils import log_sum_exp


class InvertGHMM(nn.Module):
    def __init__(self, args):
        super(InvertGHMM, self).__init__()

        self.args = args
        self.device = args.device

        # Gaussian Variance, zero
        self.gaussian_var = torch.zeros(args.n_embed,
                                        dtype=torch.float32,
                                        device=self.device,
                                        requires_grad=False)

        self.n_labels = args.n_labels
        self.couple_layers = args.couple_layers
        self.cell_layers = args.cell_layers
        self.hidden_units = args.n_embed // 2

        # transition parameters in log space
        self.transitions_params = Parameter(torch.Tensor(self.n_labels, self.n_labels))

        # Gaussian means
        self.means = Parameter(torch.Tensor(self.n_labels, self.args.n_embed))

        if args.model == 'nice':
            self.nice_layer = NICETrans(self.couple_layers,
                                        self.cell_layers,
                                        self.hidden_units,
                                        self.args.n_embed,
                                        self.device)

        self.pi = torch.zeros(self.n_labels,
                              dtype=torch.float32,
                              requires_grad=False,
                              device=self.device).fill_(1.0 / self.n_labels)

        self.pi = torch.log(self.pi)

    def init_params(self, init_seed):
        """

        Args:
            init_seed:

        Returns:

        """
        # initialize transition matrix params
        self.transitions_params.data.uniform_()

        # load pretrained model
        if self.args.load_nice != '':
            self.load_state_dict(torch.load(self.args.load_nice), strict=False)

        # load pretrained Gaussian baseline
        if self.args.load_gaussian != '':
            self.load_state_dict(torch.load(self.args.load_gaussian), strict=False)

        # initialize mean and variance with empirical values
        with torch.no_grad():
            sents, masks = init_seed
            sents, _ = self.transform(sents)
            seq_length, _, features = sents.size()
            flat_sents = sents.view(-1, features)
            seed_mean = torch.sum(masks.view(-1, 1).expand_as(flat_sents) *
                                  flat_sents, dim=0) / masks.sum()
            seed_var = torch.sum(masks.view(-1, 1).expand_as(flat_sents) *
                                 ((flat_sents - seed_mean.expand_as(flat_sents)) ** 2),
                                 dim=0) / masks.sum()
            self.gaussian_var.copy_(seed_var)
            # add noise to the pretrained Gaussian mean
            self.means.data.normal_().mul_(0.04)
            self.means.data.add_(seed_mean.data.expand_as(self.means.data))

    def _calc_log_density_c(self):
        # return -self.args.n_embed/2.0 * (math.log(2) + \
        #         math.log(np.pi)) - 0.5 * self.args.n_embed * (torch.log(self.var))

        return -self.args.n_embed / 2.0 * (math.log(2) + math.log(np.pi)) - 0.5 * torch.sum(
            torch.log(self.gaussian_var))

    def transform(self, x):
        """
        Args:
            x: (sent_length, batch_size, num_dims)
        """
        jacobian_loss = torch.zeros(1, device=self.device, requires_grad=False)

        if self.args.model == 'nice':
            x, jacobian_loss_new = self.nice_layer(x)
            jacobian_loss = jacobian_loss + jacobian_loss_new

        return x, jacobian_loss

    def unsupervised_loss(self, words, pretrained_embed, masks):
        """

        Args:
            words:
            pretrained_embed:
            masks:

        Returns:

        """
        seq_len, batch_size, _ = pretrained_embed.shape
        #
        transform_embed, jc_loss = self.transform(pretrained_embed)

        assert self.gaussian_var.data.min() > 0

        self.logA = self._calc_log_transitions()
        self.log_density_c = self._calc_log_density_c()

        density = self._eval_density(transform_embed[0])
        alpha = self.pi + density
        for t in range(1, seq_len):
            # [num_state, batch_size]
            density = self._eval_density(transform_embed[t])
            # pad mask
            mask_ep = masks[t].expand(self.n_labels, batch_size).transpose(0, 1)
            alpha = torch.mul(mask_ep, self._forward_cell(alpha, density)) + torch.mul(1 - mask_ep, alpha)

        # calculate objective from log space
        objective = torch.sum(log_sum_exp(alpha, dim=1))

        return (jc_loss - objective) / batch_size

    def _calc_alpha(self, sents, masks):
        """
        sents: (sent_length, batch_size, self.args.n_embed)
        masks: (sent_length, batch_size)
        Returns:
            output: (batch_size, sent_length, num_state)
        """
        max_length, batch_size, _ = sents.size()

        alpha_all = []
        alpha = self.pi + self._eval_density(sents[0])
        alpha_all.append(alpha.unsqueeze(1))
        for t in range(1, max_length):
            density = self._eval_density(sents[t])
            mask_ep = masks[t].expand(self.n_labels, batch_size) \
                .transpose(0, 1)
            alpha = torch.mul(mask_ep, self._forward_cell(alpha, density)) + \
                    torch.mul(1 - mask_ep, alpha)
            alpha_all.append(alpha.unsqueeze(1))

        return torch.cat(alpha_all, dim=1)

    def _forward_cell(self, alpha, density):
        batch_size = len(alpha)
        ep_size = torch.Size([batch_size, self.n_labels, self.n_labels])
        alpha = log_sum_exp(alpha.unsqueeze(dim=2).expand(ep_size) +
                            self.logA.expand(ep_size) +
                            density.unsqueeze(dim=1).expand(ep_size), dim=1)

        return alpha

    def _backward_cell(self, beta, density):
        """
        density: (batch_size, num_state)
        beta: (batch_size, num_state)
        """
        batch_size = len(beta)
        ep_size = torch.Size([batch_size, self.n_labels, self.n_labels])
        beta = log_sum_exp(self.logA.expand(ep_size) +
                           density.unsqueeze(dim=1).expand(ep_size) +
                           beta.unsqueeze(dim=1).expand(ep_size), dim=2)

        return beta

    def _eval_density(self, words):
        """
        words: (batch_size, self.args.n_embed)
        """

        batch_size = words.size(0)
        ep_size = torch.Size([batch_size, self.n_labels, self.args.n_embed])
        words = words.unsqueeze(dim=1).expand(ep_size)
        means = self.means.expand(ep_size)
        var = self.gaussian_var.expand(ep_size)

        return self.log_density_c - \
               0.5 * torch.sum((means - words) ** 2 / var, dim=2)

    def _calc_log_transitions(self):
        return self.transitions_params - log_sum_exp(self.transitions_params, dim=1, keepdim=True).expand(self.n_labels,
                                                                                                          self.n_labels)

    def _calc_log_mul_emit(self):
        return self.emission - log_sum_exp(self.emission, dim=1, keepdim=True).expand(self.n_labels, self.vocab_size)

    def predict(self, pretrained_embed, masks):
        """

        Args:
            pretrained_embed:
            masks:

        Returns:

        """
        transform_embed, jc_loss = self.transform(pretrained_embed)
        return self._viterbi(transform_embed, masks)

    def _viterbi(self, word_embed, masks):
        """
        Args:
            word_embed: (sent_length, batch_size, num_dims)
            masks: (sent_length, batch_size)
        """

        self.log_density_c = self._calc_log_density_c()
        self.logA = self._calc_log_transitions()

        length, batch_size = masks.size()

        # (batch_size, num_state)
        density = self._eval_density(word_embed[0])
        delta = self.pi + density

        ep_size = torch.Size([batch_size, self.n_labels, self.n_labels])
        index_all = []

        # forward calculate delta
        for t in range(1, length):
            density = self._eval_density(word_embed[t])
            delta_new = self.logA.expand(ep_size) + \
                        density.unsqueeze(dim=1).expand(ep_size) + \
                        delta.unsqueeze(dim=2).expand(ep_size)
            mask_ep = masks[t].view(-1, 1, 1).expand(ep_size)
            delta = mask_ep * delta_new + \
                    (1 - mask_ep) * delta.unsqueeze(dim=1).expand(ep_size)

            # index: (batch_size, num_state)
            delta, index = torch.max(delta, dim=1)
            index_all.append(index)

        assign_all = []
        # assign: (batch_size)
        _, assign = torch.max(delta, dim=1)
        assign_all.append(assign.unsqueeze(dim=1))

        # backward retrieve path
        # len(index_all) = length-1
        for t in range(length - 2, -1, -1):
            assign_new = torch.gather(index_all[t],
                                      dim=1,
                                      index=assign.view(-1, 1)).squeeze(dim=1)

            assign_new = assign_new.float()
            assign = assign.float()
            assign = masks[t + 1] * assign_new + (1 - masks[t + 1]) * assign
            assign = assign.long()

            assign_all.append(assign.unsqueeze(dim=1))

        assign_all = assign_all[-1::-1]

        return torch.cat(assign_all, dim=1)

    def save(self, path):
        """

        Args:
            path:

        Returns:

        """
        state = {
            "args": self.args,
            "state_dict": self.state_dict(),
            "var": self.gaussian_var,
            "pi": self.pi,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        """

        Args:
            path:

        Returns:

        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(path, map_location=device)
        model = cls(state["args"]).to(device)
        model.load_state_dict(state["state_dict"], strict=False)
        model.gaussian_var.data.copy_(state["var"])
        model.pi.data.copy_(state["pi"])
        return model
