gpu=3
seed=0
export CUDA_VISIBLE_DEVICES=$gpu

#for language in "de" "en" "es" "fr" "id" "it" "pt" "sv" "ko" "ja" ; do
for language in "it" ; do
  echo "====================== ${language} ========================="

  python run.py \
    --model gaussian \
    --seed $seed \
    --save "save/test" \
    --ud \
    --train_file "data/ud/${language}/total.conll" \
    --evaluate_file "data/ud/${language}/total.conll" \
    --test_file "data/ud/${language}/total.conll" \
    --word_vec "data/ml_elmo_embeddings/${language}/elmo.full.cased.1024d.txt" \

  # train entire model
  python -u run.py \
    --model nice \
    --seed $seed \
    --save "save/test" \
    --ud \
    --train_file "data/ud/${language}/total.conll" \
    --evaluate_file "data/ud/${language}/total.conll" \
    --test_file "data/ud/${language}/total.conll" \
    --word_vec "data/ml_elmo_embeddings/${language}/elmo.full.cased.1024d.txt" \
    --load_gaussian gaussian_8layers.pt

  echo ""
  echo ""
done
