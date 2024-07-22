#!/bin/bash
TOTAL_UPDATES=15000 # Total number of training steps
UPDATE_FREQ=2 # assume GPU=8
EVALB=$HOME/thirdparty/benepar/EVALB_SPMRL/evalb_spmrl
EVALB_PARAM=$HOME/thirdparty/benepar/EVALB_SPMRL/spmrl_nolabel.prm
STANFORD_GOLD=data/iwslt14_deen/tmp/stanford/test.en.tree

## bpe model ##
DATA_DIR=databin/iwslt14_deen_bpe10k
MODEL_DIR=checkpoints/iwslt14_deen_bpe10k/structformer
RESULT_DIR=experiments/iwslt_deen/results/bpe10k/structformer
mkdir -p $RESULT_DIR

rm -rf $MODEL_DIR

poetry run fairseq-train $DATA_DIR \
    --user-dir src \
    --arch structformer_iwslt_de_en --share-decoder-input-output-embed \
    --induction-layer 1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe --quiet


# token model ##
DATA_DIR=databin/iwslt14_deen_nobpe
MODEL_DIR=checkpoints/iwslt14_deen_nobpe/structformer
RESULT_DIR=experiments/iwslt_deen/results/nobpe/structformer
mkdir -p $RESULT_DIR

rm -rf $MODEL_DIR

poetry run fairseq-train $DATA_DIR \
    --user-dir src \
    --arch structformer_iwslt_de_en --share-decoder-input-output-embed \
    --induction-layer 1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES

poetry run fairseq-generate $DATA_DIR \
    --user-dir src \
    --gen-subset test \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --quiet

poetry run python generate_grammar.py $DATA_DIR \
    --gen-subset test \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 1 \
    --results-path $RESULT_DIR

grep ^TT- $RESULT_DIR/generate-test.txt | sort -V | cut -f2 > $RESULT_DIR/pred.orig.tree
# postprocess to fill in the <unk>
poetry run python complete_tree.py -t data/iwslt14_deen/tmp/test.en < $RESULT_DIR/pred.orig.tree > $RESULT_DIR/pred.cplt.tree

paste $RESULT_DIR/pred.cplt.tree $STANFORD_GOLD | grep -v "|||" > $RESULT_DIR/tmp.tsv
cut -f1 $RESULT_DIR/tmp.tsv > $RESULT_DIR/pred.tree
cut -f2 $RESULT_DIR/tmp.tsv > $RESULT_DIR/gold.tree

$EVALB -p $EVALB_PARAM $RESULT_DIR/gold.tree $RESULT_DIR/pred.tree
