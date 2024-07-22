#!/bin/bash
TOTAL_UPDATES=100000   # Total number of training steps
UPDATE_FREQ=2          # 8 GPUs

FAIRSEQ_DIR=$HOME/github_project/fairseq
DATA_DIR=databin/wmt16_ende_bpe32k
MODEL_DIR=checkpoints/wmt16_ende_bpe32k/transformer_sync
RESULT_DIR=experiments/wmt_deen/results/bpe32k/transformer_sync
mkdir -p $RESULT_DIR

pushd $HOME/fairseq_project/translation_with_synchronous_grammar

# poetry run fairseq-train $DATA_DIR --fp16 \
#     --user-dir $HOME/fairseq_project/nmt_ssa/src \
#     --arch transformer_sync_vaswani_wmt_en_de_big --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
#     --dropout 0.3 --weight-decay 0.0  \
#     --criterion label_smoothed_cross_entropy_with_sync --label-smoothing 0.1 --sync-lambda 1.0 \
#     --max-tokens 3584 --update-freq $UPDATE_FREQ \
#     --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

fairseq-generate $DATA_DIR \
    --user-dir $HOME/fairseq_project/nmt_ssa/src \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 4 --lenpen 0.6 --remove-bpe \
    --results-path $RESULT_DIR

bash $FAIRSEQ_DIR/scripts/sacrebleu.sh wmt14/full en de $RESULT_DIR/generate-test.txt

popd
