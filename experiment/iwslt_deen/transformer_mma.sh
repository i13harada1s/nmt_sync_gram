#!/bin/bash
DATA_DIR=databin/wat18_jaen_bpe100k
RESULT_DIR=experiments/results/wat18_jaen_bpe100k
FAIRSEQ_DIR=$HOME/fairseq
SCRIPTS_DIR=scripts

mkdir -p $RESULT_DIR

TOTAL_UPDATES=100 #50000   # Total number of training steps
UPDATE_FREQ=2

#####################
# Translation Task
#####################

# Japanese -> English

MODEL_DIR=checkpoints/wat18_jaen/transformer_ja_en
rm -rf $MODEL_DIR

pipenv run fairseq-train $DATA_DIR \
    --arch transformer --share-decoder-input-output-embed \
    --task translation \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

# pipenv run fairseq-generate $DATA_DIR \
#     --user-dir src \
#     --path $MODEL_DIR/checkpoint_best.pt \
#     --batch-size 128 --beam 5 --remove-bpe \
#     --results-path $RESULT_DIR

# grep ^H $RESULT_DIR/generate-test.txt \
#     | sed 's/^H\-//' \
#     | sort -n -k 1 \
#     | cut -f 3 \
#     | perl -C -pe 's/([^Ａ-Ｚａ-ｚA-Za-z]) +/${1}/g; s/ +([^Ａ-Ｚａ-ｚA-Za-z])/${1}/g;' \
#     > $RESULT_DIR/transformer.test.ja2en.sys.sorted

# pipenv run sacrebleu --test-set data/wat18_jaen_100k/tmp/test.en --language-pair "ja-en" < $RESULT_DIR/transformer.test.ja2en.sys.sorted
