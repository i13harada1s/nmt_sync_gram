#!/bin/bash
RESULT_DIR=experiments/results/align_roen_spm32k
mkdir -p $RESULT_DIR

TOTAL_UPDATES=50000    # Total number of training steps

SRC=ro
TGT=en

#####################
# Translation Task
#####################

DATA_DIR=databin/align_roen_spm32k
UPDATE_FREQ=2

# Russian -> English
MODEL_DIR=checkpoints/align_roen_spm32k/structformer_rank_${SRC}_${TGT}

poetry run fairseq-train $DATA_DIR \
    --user-dir src --source-lang $SRC --target-lang $TGT \
    --arch structformer_align_chen2020 --share-all-embeddings \
    --induction-layer 1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_sync --label-smoothing 0.1 \
    --sync-lambda 0.01 --sync-method "rank" --hinge-margen 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece --quiet 

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --source-lang $SRC --target-lang $TGT \
    --score-reference --print-alignment hard \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 512 --beam 1 --remove-bpe=sentencepiece \
    --results-path $RESULT_DIR

grep ^A $RESULT_DIR/generate-test.txt | sort -V | cut -f2 > $RESULT_DIR/structformer_rank.${SRC}2${TGT}.spm.talp

# English -> Russian

MODEL_DIR=checkpoints/align_roen_spm32k/structformer_rank_${TGT}_${SRC}

poetry run fairseq-train $DATA_DIR \
    --user-dir src --source-lang $TGT --target-lang $SRC \
    --arch structformer_align_chen2020 --share-all-embeddings \
    --induction-layer 1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_sync --label-smoothing 0.1 \
    --sync-lambda 0.01 --sync-method "rank" --hinge-margen 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece --quiet 

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --source-lang $SRC --target-lang $TGT \
    --score-reference --print-alignment hard \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 512 --beam 1 --remove-bpe=sentencepiece \
    --results-path $RESULT_DIR

grep ^A $RESULT_DIR/generate-test.txt | sort -V | cut -f2 > $RESULT_DIR/structformer_rank.${TGT}2${SRC}.spm.talp

#####################
# Alignment Task
#####################

DATA_DIR=data/align_roen_spm32k
ALIGN_SCRIPTS=$HOME/thirdparty/alignment-scripts

SPM2TKN=$ALIGN_SCRIPTS/scripts/sentencepiece_to_word_alignments.py
COMBINE=$ALIGN_SCRIPTS/scripts/combine_bidirectional_alignments.py
EVALUATE=$ALIGN_SCRIPTS/scripts/aer.py
GOLD_ALIGN=$ALIGN_SCRIPTS/test/${SRC}${TGT}.talp

echo "Converting bpe-level alignments to token-level alignments: $RESULT_DIR/structformer_rank.${SRC}2${TGT}.spm.talp -> $RESULT_DIR/structformer_rank.${SRC}2${TGT}.talp"
poetry run python $SPM2TKN $DATA_DIR/test.$SRC $DATA_DIR/test.$TGT < $RESULT_DIR/structformer_rank.${SRC}2${TGT}.spm.talp > $RESULT_DIR/structformer_rank.${SRC}2${TGT}.talp
# poetry run python $SPM2TKN $DATA_DIR/test.$TGT $DATA_DIR/test.$SRC < $RESULT_DIR/structformer_rank.${SRC}2${TGT}.spm.talp > $RESULT_DIR/structformer_rank.${SRC}2${TGT}.talp

echo "Converting bpe-level alignments to token-level alignments: $RESULT_DIR/structformer_rank.${TGT}2${SRC}.spm.talp -> $RESULT_DIR/structformer_rank.${TGT}2${SRC}.talp"
poetry run python $SPM2TKN $DATA_DIR/test.$TGT $DATA_DIR/test.$SRC < $RESULT_DIR/structformer_rank.${TGT}2${SRC}.spm.talp > $RESULT_DIR/structformer_rank.${TGT}2${SRC}.talp
# poetry run python $SPM2TKN $DATA_DIR/test.$SRC $DATA_DIR/test.$TGT < $RESULT_DIR/structformer_rank.${TGT}2${SRC}.spm.talp > $RESULT_DIR/structformer_rank.${TGT}2${SRC}.talp

echo "Combining bidirectional token-level alignments: ($RESULT_DIR/structformer_rank.${SRC}2${TGT}.talp, $RESULT_DIR/structformer_rank.${TGT}2${SRC}.talp) -> $RESULT_DIR/structformer_rank.${SRC}${TGT}.talp"
poetry run python $COMBINE $RESULT_DIR/structformer_rank.${SRC}2${TGT}.talp $RESULT_DIR/structformer_rank.${TGT}2${SRC}.talp --method "grow-diagonal" > $RESULT_DIR/structformer_rank.${SRC}${TGT}.talp

echo "Evaluate $RESULT_DIR/structformer_rank.${SRC}2${TGT}.talp"
poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/structformer_rank.${SRC}2${TGT}.talp --fAlpha 0.5 --oneRef 

echo "Evaluate $RESULT_DIR/structformer_rank.${TGT}2${SRC}.talp"
poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/structformer_rank.${TGT}2${SRC}.talp --fAlpha 0.5 --oneRef --reverseHyp

echo "Evaluate $RESULT_DIR/structformer_rank.${SRC}${TGT}.talp"
poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/structformer_rank.${SRC}${TGT}.talp --fAlpha 0.5 --oneRef 
