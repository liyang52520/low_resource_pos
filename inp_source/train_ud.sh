num=4
gpu=$num
seed=$num
export CUDA_VISIBLE_DEVICES=$gpu

for language in "de" "en" "es" "fr" "id" "it" "pt" "sv" "ko" "ja" ; do
  echo "====================== ${language} ========================="

  python run.py \
    --model gaussian \
    --seed $seed \
    --save "save/${language}_${seed}" \
    --ud \
    --train_file "data/ud/${language}/${language}-universal-train.conll" \
    --evaluate_file "data/ud/${language}/${language}-universal-dev.conll" \
    --test_file "data/ud/${language}/${language}-universal-test.conll" \
    --word_vec "data/ml_elmo_embeddings/${language}/elmo.full.cased.1024d.txt" \

  # train entire model
  python -u run.py \
    --model nice \
    --seed $seed \
    --save "save/${language}_${seed}" \
    --ud \
    --train_file "data/ud/${language}/${language}-universal-train.conll" \
    --evaluate_file "data/ud/${language}/${language}-universal-dev.conll" \
    --test_file "data/ud/${language}/${language}-universal-test.conll" \
    --word_vec "data/ml_elmo_embeddings/${language}/elmo.full.cased.1024d.txt" \
    --load_gaussian gaussian_8layers.pt

  echo ""
  echo ""
done
