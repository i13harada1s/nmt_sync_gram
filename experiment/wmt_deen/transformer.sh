#!/bin/bash
TOTAL_UPDATES=100000   # Total number of training steps
UPDATE_FREQ=2          # 8 GPUs

pushd $HOME/fairseq_project/translation_with_synchronous_grammar

FAIRSEQ_DIR=$HOME/github_project/fairseq
DATA_DIR=databin/wmt16_ende_bpe32k
MODEL_DIR=checkpoints/wmt16_ende_bpe32k/transformer
RESULT_DIR=experiments/wmt_deen/results/bpe32k/transformer
mkdir -p $RESULT_DIR

fairseq-train $DATA_DIR --fp16 \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 --update-freq $UPDATE_FREQ \
    --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

# --keep-last-epochs 10 
# python scripts/average_checkpoints \
#     --inputs $MODEL_DIR \
#     --num-epoch-checkpoints 10 \
#     --output $MODEL_DIR/checkpoint.avg10.pt

fairseq-generate $DATA_DIR \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 4 --lenpen 0.6 --remove-bpe \
    --results-path $RESULT_DIR

bash $FAIRSEQ_DIR/scripts/sacrebleu.sh wmt14/full en de $RESULT_DIR/generate-test.txt

# grep ^H $RESULT_DIR/generate-test.txt \
#     | sed 's/^H\-//' \
#     | sort -n -k 1 \
#     | cut -f 3 \
#     | sacremoses detokenize \
#     > $RESULT_DIR/transformer.test.en2de.sys.sorted.detok

# sacrebleu ~/data/wmt16_en_de_bpe32k/newstest2014.de -i $RESULT_DIR/transformer.test.en2de.sys.sorted.detok --language-pair "de-en" -w 2 

popd
