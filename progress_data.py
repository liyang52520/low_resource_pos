import os.path
import random


def unify_data_style(base_dir):
    src_dir = os.path.join("data/corpus", base_dir)
    tgt_dir = os.path.join("data/new_corpus", base_dir)
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)
    for file_name in os.listdir(src_dir):
        src_file_path = os.path.join(src_dir, file_name)
        tgt_file_path = os.path.join(tgt_dir, file_name)
        print(src_file_path, "===>", tgt_file_path)

        src_file = open(src_file_path, encoding="utf8")
        tgt_file = open(tgt_file_path, "w", encoding="utf8")

        for line in src_file:
            line = line.strip()
            if len(line) > 0:
                idx, word, _, pos, *_ = line.split("\t")
                # idx, word, _, _, pos, *_ = line.split("\t")
                print(f"{idx}\t{word}\t_\t_\t{pos}\t_\t_\t_\t_\t_", file=tgt_file)
            else:
                print(file=tgt_file)

        src_file.close()
        tgt_file.close()


def random_select_samples(base_dir, sample_count, random_seed=0):
    """

    Args:
        base_dir
        sample_count:
        random_seed:

    Returns:

    """
    random.seed(random_seed)
    file_path = os.path.join(base_dir, "train.conll")
    sentences = []
    full_label_set = set()
    with open(file_path, encoding="utf8") as f:
        sentence = [[], [], [], 0]
        sentence_idx = 0
        for line in f:
            if len(line) > 1:
                idx, word, _, _, label, *_ = line.split("\t")
                sentence[0].append(idx)
                sentence[1].append(word)
                sentence[2].append(label)
                full_label_set.add(label)
            else:
                sentence[-1] = sentence_idx
                sentence_idx += 1
                sentences.append(sentence)
                sentence = [[], [], [], 0]
    # random select sentences
    selected_idxes = random.sample(list(range(len(sentences))), sample_count)
    #
    selected_sentences = [
        sentences[selected_idx]
        for selected_idx in selected_idxes
    ]
    #
    selected_label_set = set(sum([
        sentence[2]
        for sentence in selected_sentences
    ], []))

    # choose several sentences in addition to assure that all labels appear
    while len(selected_label_set) != len(full_label_set):
        sorted_sentences = sorted(sentences,
                                  key=lambda x: len((full_label_set - selected_label_set) & set(x[2])),
                                  reverse=True)
        for sentence in sorted_sentences:
            if sentence[-1] not in selected_idxes:
                selected_sentences.append(sentence)
                selected_idxes.append(sentence[-1])
                selected_label_set = selected_label_set.union(set(sentence[2]))
                break
    # out file
    tgt_file = open(os.path.join(base_dir, f"train.{sample_count}.conll"), "w", encoding="utf8")

    print(f"{len(selected_sentences)} sentences selected!")
    for sentence in selected_sentences:
        for idx, word, label in zip(*sentence[:-1]):
            print(f"{idx}\t{word}\t_\t_\t{label}\t_\t_\t_\t_\t_", file=tgt_file)
        print(file=tgt_file)

    tgt_file.close()


if __name__ == '__main__':
    ptb_sentence_count = 0
    with open("data/corpus/ptb/train.conll", encoding="utf8") as f:
        for line in f:
            if len(line) > 1:
                pass
            else:
                ptb_sentence_count += 1
    print(ptb_sentence_count)
    for lang in os.listdir("data/corpus/ud"):
        print(lang)
        lang_sentence_count = 0
        with open(f"data/corpus/ud/{lang}/train.conll", encoding="utf8") as f:
            for line in f:
                if len(line) > 1:
                    pass
                else:
                    lang_sentence_count += 1
        print(lang_sentence_count)
        for c in (int(lang_sentence_count * 100.0 / ptb_sentence_count),):
            print(c)
            random_select_samples(f"data/corpus/ud/{lang}", c)
