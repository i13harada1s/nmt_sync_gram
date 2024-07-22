#!/bin/bash
RESULT_DIR=experiments/results/align_deen_spm32k
mkdir -p $RESULT_DIR

TOTAL_UPDATES=50000    # 50K

#####################
# Translation Task
#####################

DATA_DIR=databin/align_deen_spm32k

MAX_TOKENS=4096
UPDATE_FREQ=2

SRC=de
TGT=en

# Garmman -> English

MODEL_DIR=checkpoints/align_deen_spm32k/structformer_mse_de_en
rm -rf $MODEL_DIR

fairseq-train $DATA_DIR \
    --user-dir src --source-lang de --target-lang en \
    --arch structformer_align_chen2020 --share-all-embeddings \
    --inductor-layers 1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001  \
    --criterion label_smoothed_cross_entropy_with_sync --label-smoothing 0.1 \
    --sync-lambda 0.05 --sync-method "mse" \
    --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

fairseq-generate $DATA_DIR \
    --user-dir src --source-lang de --target-lang en \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe=sentencepiece --quiet

# NOTE: Even if set --remove-bpe, the system outputs the alignment at the bpe-level.
fairseq-generate $DATA_DIR \
    --user-dir src --source-lang de --target-lang en \
    --score-reference --print-alignment \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --remove-bpe=sentencepiece \
    --results-path $RESULT_DIR

grep ^A $RESULT_DIR/generate-test.txt | sort -V | cut -f2 > $RESULT_DIR/structformer_mse.${SRC}2${TGT}.spm.talp

# # English -> Garmman

# MODEL_DIR=checkpoints/align_deen_spm32k/structformer_mse_en_de
# rm -rf $MODEL_DIR

# poetry run fairseq-train $DATA_DIR \
#     --user-dir src --source-lang en --target-lang de \
#     --arch structformer_align_chen2020 --share-all-embeddings \
#     --inductor-layers 1 \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001  \
#     --criterion label_smoothed_cross_entropy_with_sync --label-smoothing 0.1 \
#     --sync-lambda 0.01 --sync-method "mse" \
#     --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
#     --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

# poetry run fairseq-generate $DATA_DIR \
#     --user-dir src --source-lang en --target-lang de \
#     --path $MODEL_DIR/checkpoint_best.pt \
#     --batch-size 128 --beam 5 --remove-bpe=sentencepiece --quiet

# poetry run fairseq-generate $DATA_DIR \
#     --user-dir src --source-lang en --target-lang de \
#     --score-reference --print-alignment \
#     --path $MODEL_DIR/checkpoint_best.pt \
#     --batch-size 128 --remove-bpe=sentencepiece \
#     --results-path $RESULT_DIR

# grep ^A $RESULT_DIR/generate-test.txt | sort -V | cut -f2 > $RESULT_DIR/structformer_mse.${TGT}2${SRC}.spm.talp

# #####################
# # Alignment Task
# #####################

# RAW_DATA_DIR=data/align_deen_spm32k
# ALIGN_SCRIPTS=$HOME/thirdparty/alignment-scripts

# SPM2TKN=$ALIGN_SCRIPTS/scripts/sentencepiece_to_word_alignments.py
# COMBINE=$ALIGN_SCRIPTS/scripts/combine_bidirectional_alignments.py
# EVALUATE=$ALIGN_SCRIPTS/scripts/aer.py
# GOLD_ALIGN=$ALIGN_SCRIPTS/test/deen.talp

# echo "Converting bpe-level alignments to token-level alignments: $RESULT_DIR/structformer_mse.de2en.spm.talp -> $RESULT_DIR/structformer_mse.de2en.talp"
# poetry run python $SPM2TKN $RAW_DATA_DIR/test.de $RAW_DATA_DIR/test.en < $RESULT_DIR/structformer_mse.de2en.spm.talp > $RESULT_DIR/structformer_mse.de2en.talp

# echo "Converting bpe-level alignments to token-level alignments: $RESULT_DIR/structformer_mse.en2de.spm.talp -> $RESULT_DIR/structformer_mse.en2de.talp"
# poetry run python $SPM2TKN $RAW_DATA_DIR/test.en $RAW_DATA_DIR/test.de < $RESULT_DIR/structformer_mse.en2de.spm.talp > $RESULT_DIR/structformer_mse.en2de.talp

# echo "Combining bidirectional token-level alignments: ($RESULT_DIR/structformer_mse.de2en.talp, $RESULT_DIR/structformer_mse.en2de.talp) -> $RESULT_DIR/structformer_mse.deen.talp"
# poetry run python $COMBINE $RESULT_DIR/structformer_mse.de2en.talp $RESULT_DIR/structformer_mse.en2de.talp --method "grow-diagonal" > $RESULT_DIR/structformer_mse.deen.talp

# echo "Evaluate $RESULT_DIR/structformer_mse.de2en.talp"
# poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/structformer_mse.de2en.talp --fAlpha 0.5 --oneRef 

# echo "Evaluate $RESULT_DIR/structformer_mse.en2de.talp"
# poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/structformer_mse.en2de.talp --fAlpha 0.5 --oneRef --reverseHyp

# echo "Evaluate $RESULT_DIR/structformer_mse.deen.talp"
# poetry run python $EVALUATE $GOLD_ALIGN $RESULT_DIR/structformer_mse.deen.talp --fAlpha 0.5 --oneRef 
