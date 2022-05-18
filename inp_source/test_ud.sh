gpu=3
seed=0
export CUDA_VISIBLE_DEVICES=$gpu

#for language in "de" "en" "es" "fr" "id" "it" "pt" "sv" "ko" "ja" ; do
for language in "de" ; do
  echo "====================== ${language} ========================="

#  python run.py \
#    --model gaussian \
#    --seed $seed \
#    --save "save/test" \
#    --ud \
#    --train_file "data/ud/${language}/${language}-universal-train.conll" \
#    --evaluate_file "data/ud/${language}/${language}-universal-dev.conll" \
#    --test_file "data/ud/${language}/${language}-universal-test.conll" \
#    --word_vec "data/fast_text/fast_text.${language}.300d.txt" \

  # train entire model
  python -u run.py \
    --model nice \
    --seed $seed \
    --save "save/test" \
    --ud \
    --train_file "data/ud/${language}/${language}-universal-train.conll" \
    --evaluate_file "data/ud/${language}/${language}-universal-dev.conll" \
    --test_file "data/ud/${language}/${language}-universal-test.conll" \
    --word_vec "data/fast_text/fast_text.${language}.300d.txt" \
    --load_gaussian gaussian_8layers.pt

  echo ""
  echo ""
done
