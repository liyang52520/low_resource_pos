model="crf_ae"
python run.py train \
  --device 7 \
  --model $model \
  --config "configs/${model}.ini" \
  --train data/corpus/ptb/train.conll \
  --train-labeled data/corpus/ptb/train.100.conll \
  --dev data/corpus/ptb/dev.conll \
  --test data/corpus/ptb/test.conll \
  --embed data/embed/fasttext/fasttext.en.300d.txt \
  --save save/test
