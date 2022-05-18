gpu=5
seed=0
export CUDA_VISIBLE_DEVICES=$gpu

#python run.py \
#  --model gaussian \
#  --seed $seed \
#  --save "save/test" \
#  --train_file "data/invert/train.conll" \
#  --evaluate_file "data/invert/dev.conll" \
#  --test_file "data/invert/test.conll" \
#  --word_vec data/wsj_word_vec.pkl

# train entire model
python -u run.py \
  --model nice \
  --seed $seed \
  --save "save/test" \
  --train_file "data/invert/train.conll" \
  --evaluate_file "data/invert/dev.conll" \
  --test_file "data/invert/test.conll" \
  --word_vec data/wsj_word_vec.pkl \
  --load_gaussian "gaussian_8layers.pt"
