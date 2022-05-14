python run.py train \
  --device 4 \
  --model crf \
  --save save/test \
  --train data/corpus/ptb/train.conll \
  --train-labeled data/corpus/ptb/train.100.conll \
  --dev data/corpus/ptb/dev.conll \
  --test data/corpus/ptb/test.conll \
  --embed data/embed/fasttext/fasttext.en.300d.txt \
