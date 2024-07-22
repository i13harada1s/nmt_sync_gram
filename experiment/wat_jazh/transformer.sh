#!/bin/bash
TOTAL_UPDATES=100000   # Total number of training steps
UPDATE_FREQ=2

# pushd $HOME/fairseq_project/translation_with_synchronous_grammar

DATA_DIR=databin/wat18_jazh
MODEL_DIR=checkpoints/wat18_jazh/transformer
RESULT_DIR=experiments/wat_jazh/results/ja2zh/transformer
FAIRSEQ_DIR=$HOME/fairseq

mkdir -p $RESULT_DIR

fairseq-train $DATA_DIR --fp16 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

fairseq-generate $DATA_DIR \
    --user-dir src \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --results-path $RESULT_DIR

tail -10 $RESULT_DIR/generate-test.txt

# grep ^H $RESULT_DIR/generate-test.txt \
#     | sed 's/^H\-//' \
#     | sort -n -k 1 \
#     | cut -f 3 \
#     | perl -C -pe 's/([^Ａ-Ｚａ-ｚA-Za-z]) +/${1}/g; s/ +([^Ａ-Ｚａ-ｚA-Za-z])/${1}/g;' \
#     > $RESULT_DIR/transformer.test.ja2en.sys.sorted

# pipenv run sacrebleu --test-set data/wat18_jazh_100k/tmp/test.en --language-pair "ja-en" < $RESULT_DIR/transformer.test.ja2en.sys.sorted

# popd
