#!/bin/bash
TOTAL_UPDATES=15000    # Total number of training steps
UPDATE_FREQ=2 # assume GPU=8

# ## bpe model ##
DATA_DIR=databin/iwslt14_deen_bpe10k
MODEL_DIR=checkpoints/iwslt14_deen_bpe10k/transformer
RESULT_DIR=experiments/iwslt_deen/results/bpe10k/transformer
mkdir -p $RESULT_DIR

pipenv run fairseq-train $DATA_DIR \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

pipenv run fairseq-generate $DATA_DIR \
    --user-dir src --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe --quiet

## token model ##
DATA_DIR=databin/iwslt14_deen_nobpe
MODEL_DIR=checkpoints/iwslt14_deen_nobpe/transformer
RESULT_DIR=experiments/iwslt_deen/results/nobpe/transformer
mkdir -p $RESULT_DIR

pipenv run fairseq-train $DATA_DIR \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

pipenv run fairseq-generate $DATA_DIR \
    --user-dir src --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --quiet
