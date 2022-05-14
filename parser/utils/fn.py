import torch


def register_grad_hook(t):
    """

    Args:
        t (tensor):

    Returns:

    """
    if t.requires_grad:
        t.register_hook(lambda grad: grad.masked_fill_(torch.isnan(grad), 0))


def pad(tensors, padding_value=0, total_length=None, padding_side='right'):
    size = [len(tensors)] + [
        max(tensor.size(i) for tensor in tensors)
        for i in range(len(tensors[0].size()))
    ]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[
            slice(-i, None) if padding_side == 'left' else slice(0, i)
            for i in tensor.size()
        ]] = tensor
    return out_tensor


def compute_label_scores(emits, transitions, start, end, labels, mask, variable=False):
    """

    Args:
        emits:
        transitions:
        start:
        end:
        labels:
        mask:
        variable:

    Returns:

    """
    batch_size, seq_len = mask.shape
    scores = torch.gather(emits[mask], -1, labels[mask].unsqueeze(-1)).sum()
    # start and end
    if variable:
        scores += torch.gather(start, -1, labels[:, :1]).sum()
        batch = torch.arange(batch_size, dtype=torch.long).to(emits.device)
        scores += torch.gather(end, -1, labels[batch, mask.sum(-1) - 1].unsqueeze(-1)).sum()
    else:
        scores += start[labels[:, 0]].sum()
        batch = torch.arange(batch_size, dtype=torch.long).to(emits.device)
        scores += end[labels[batch, mask.sum(-1) - 1]].sum()
    # transitions
    for i in range(1, seq_len):
        if variable:
            scores += transitions[:, i - 1][batch, labels[:, i - 1], labels[:, i]][mask[:, i]].sum()
        else:
            scores += transitions[labels[:, i - 1], labels[:, i]][mask[:, i]].sum()
    return scores


def compute_log_z(emits, transitions, start, end, mask, use_max=False, reduction="sum", variable=False):
    """

    Args:
        emits:
        transitions: [batch_size, seq_len - 1, n_labels, n_labels]
        start:
        end:
        mask:
        use_max:
        reduction:
        variable:

    Returns:

    """
    batch_size, seq_len, n_labels = emits.shape
    if variable:
        # [batch_size, n_labels] + [batch_size, n_labels]
        log_score = start + emits[:, 0]
    else:
        log_score = start.unsqueeze(0) + emits[:, 0]
    for i in range(1, seq_len):
        if variable:
            # [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels] => [batch_size, n_labels, n_labels]
            score = log_score.unsqueeze(-1) + transitions[:, i - 1] + emits[:, i].unsqueeze(1)
        else:
            # [batch_size, n_labels, 1] + [1, n_labels, n_labels] + [batch_size, 1, n_labels]
            score = log_score.unsqueeze(-1) + transitions.unsqueeze(0) + emits[:, i].unsqueeze(1)
        if not use_max:
            log_score[mask[:, i]] = torch.logsumexp(score, dim=1)[mask[:, i]]
        else:
            temp, _ = torch.max(score, dim=1)
            log_score[mask[:, i]] = temp[mask[:, i]]
    if variable:
        log_score = log_score + end
    else:
        log_score = log_score + end.unsqueeze(0)

    # res
    if not use_max:
        log_score = torch.logsumexp(log_score, dim=-1)
    else:
        log_score, _ = torch.max(log_score, dim=-1)
    if reduction == "sum":
        log_score = log_score.sum()
    elif reduction == "average":
        log_score = log_score / batch_size
    return log_score


def viterbi(emits, transitions, start, end, mask, variable=False):
    """

    Args:
        emits: [batch_size, seq_len, n_labels]
        transitions: [batch_size, seq_len - 1, n_labels, n_labels] or [n_labels, n_labels]
        start: [batch_size, n_labels] or [n_labels]
        end: [batch_size, n_labels] or [n_labels]
        mask: [batch_size, seq_len]
        variable:

    Returns:

    """
    batch_size, seq_len, n_labels = emits.shape

    last_next_pos = mask.sum(1)

    # [batch_size, seq_len + 1, n_labels]
    path = emits.new_zeros((batch_size, seq_len + 1, n_labels)).long()

    # start
    # [batch_size, n_labels]
    if variable:
        score = start + emits[:, 0]
    else:
        score = start.unsqueeze(0) + emits[:, 0]

    for i in range(1, seq_len):
        if variable:
            # [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels] => [batch_size, n_labels, n_labels]
            temp_score = score.unsqueeze(-1) + transitions[:, i - 1] + emits[:, i].unsqueeze(1)
        else:
            # [batch_size, n_labels, 1] + [1, n_labels, n_labels] => [batch_size, n_labels, n_labels]
            temp_score = score.unsqueeze(-1) + transitions.unsqueeze(0) + emits[:, i].unsqueeze(1)
        # [batch_size, n_labels]
        temp_score, path[:, i] = torch.max(temp_score, dim=1)
        score[mask[:, i]] = temp_score[mask[:, i]]
        path[:, i][~mask[:, i]] = 0

    # end
    if variable:
        score = score + end
    else:
        score = score + end.unsqueeze(0)
    batch = torch.arange(batch_size, dtype=torch.long).to(emits.device)
    path[batch, last_next_pos, 0] = torch.argmax(score, dim=-1)

    # tags: [batch_size, seq_len]
    tags = emits.new_zeros((batch_size, seq_len)).long()
    # pre_tags: [batch_size, 1]
    pre_tags = emits.new_zeros((batch_size, 1)).long()
    for i in range(seq_len, 0, -1):
        j = i - seq_len - 1
        # pre_tags: [batch_size, 1]
        pre_tags = torch.gather(path[:, i], 1, pre_tags)
        tags[:, j] = pre_tags.squeeze()
    return tags
