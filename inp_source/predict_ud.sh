num=0
gpu=$num
seed=$num
export CUDA_VISIBLE_DEVICES=$gpu

language="de"

# train entire model
python -u run.py \
  --model gaussian \
  --seed $seed \
  --save "save/de_${seed}" \
  --ud \
  --train_file "data/ud/${language}/${language}-universal-train.conll" \
  --evaluate_file "data/ud/${language}/${language}-universal-dev.conll" \
  --test_file "data/ud/${language}/${language}-universal-test.conll" \
  --word_vec "data/ml_elmo_embeddings/${language}/elmo.full.cased.1024d.txt" \
  --load_gaussian gaussian_8layers.pt
