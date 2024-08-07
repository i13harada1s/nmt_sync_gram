#!/bin/bash
TOTAL_UPDATES=50000

#####################
# Translation Task
#####################

DATA_DIR=databin/align_jaen_spm32k
UPDATE_FREQ=2

SRC=ja
TGT=en


RESULT_DIR=experiments/align_${SRC}${TGT}/results/spm32k/transformer

# Garmman -> English

MODEL_DIR=checkpoints/align_jaen_spm32k/transformer_${SRC}_${TGT}
mkdir -p $RESULT_DIR

poetry run fairseq-train $DATA_DIR \
    --user-dir src --source-lang $SRC --target-lang $TGT \
    --arch transformer_align_chen2020 --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --source-lang $SRC --target-lang $TGT \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece --quiet

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --source-lang $SRC --target-lang $TGT \
    --score-reference --print-alignment hard \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --remove-bpe=sentencepiece \
    --results-path $RESULT_DIR

grep ^A $RESULT_DIR/generate-test.txt | sort -V | cut -f2 > $RESULT_DIR/transformer.${SRC}2${TGT}.spm.talp


# English -> Garmman

MODEL_DIR=checkpoints/align_jaen_spm32k/transformer_${TGT}_${SRC}
mkdir -p $RESULT_DIR

poetry run fairseq-train $DATA_DIR \
    --user-dir src --source-lang $TGT --target-lang $SRC \
    --arch transformer_align_chen2020 --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --source-lang $TGT --target-lang $SRC \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece --quiet

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --source-lang $TGT --target-lang $SRC \
    --score-reference --print-alignment hard \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --remove-bpe=sentencepiece \
    --results-path $RESULT_DIR

grep ^A $RESULT_DIR/generate-test.txt | sort -V | cut -f2 > $RESULT_DIR/transformer.${TGT}2${SRC}.spm.talp


#####################
# Alignment Task
#####################

RAW_DATA_DIR=data/align_jaen_spm32k

ALIGN_SCRIPTS=$HOME/thirdparty/alignment-scripts
SPM2TKN=$ALIGN_SCRIPTS/scripts/sentencepiece_to_word_alignments.py
COMBINE=$ALIGN_SCRIPTS/scripts/combine_bidirectional_alignments.py
EVALUATE=$ALIGN_SCRIPTS/scripts/aer.py
GOLD_ALIGN=$HOME/data/kftt-alignments/data/align-test.txt

echo "Converting bpe-level alignments to token-level alignments: $RESULT_DIR/transformer.${SRC}2${TGT}.spm.talp -> $RESULT_DIR/transformer.${SRC}2${TGT}.talp"
poetry run python $SPM2TKN $RAW_DATA_DIR/test.${SRC} $RAW_DATA_DIR/test.${TGT} < $RESULT_DIR/transformer.${SRC}2${TGT}.spm.talp > $RESULT_DIR/transformer.${SRC}2${TGT}.talp

echo "Converting bpe-level alignments to token-level alignments: $RESULT_DIR/transformer.${TGT}2${SRC}.spm.talp -> $RESULT_DIR/transformer.${TGT}2${SRC}.talp"
poetry run python $SPM2TKN $RAW_DATA_DIR/test.${TGT} $RAW_DATA_DIR/test.${SRC} < $RESULT_DIR/transformer.${TGT}2${SRC}.spm.talp > $RESULT_DIR/transformer.${TGT}2${SRC}.talp

echo "Combining bidirectional token-level alignments: ($RESULT_DIR/transformer.${SRC}2${TGT}.talp, $RESULT_DIR/transformer.${TGT}2${SRC}.talp) -> $RESULT_DIR/transformer.jaen.talp"
poetry run python $COMBINE $RESULT_DIR/transformer.${SRC}2${TGT}.talp $RESULT_DIR/transformer.${TGT}2${SRC}.talp --method "grow-diagonal" > $RESULT_DIR/transformer.jaen.talp

echo "Evaluate $RESULT_DIR/transformer.${SRC}2${TGT}.talp"
poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/transformer.${SRC}2${TGT}.talp --fAlpha 0.5

echo "Evaluate $RESULT_DIR/transformer.${TGT}2${SRC}.talp"
poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/transformer.${TGT}2${SRC}.talp --fAlpha 0.5 --reverseHyp

echo "Evaluate $RESULT_DIR/transformer.${SRC}${TGT}.talp"
poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/transformer.${SRC}${TGT}.talp --fAlpha 0.5
