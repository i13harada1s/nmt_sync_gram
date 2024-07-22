# Neural Machine Translation with Synchronous Latent Phrase Structure (Harada et al., 2021)

## Introduction
This is an implementation of "[Neural Machine Translation with Synchronous Latent Phrase Structur (Harada et al., 2021)](https://aclanthology.org/2021.acl-srw.33/)".


## Getting Started:
The documentation consists of two instructions: [preparation](preparation/README.md) and [training](experiment/README.md).

### Environment
Python environment can be build and run with Dockerfile:
```sh
cd nmt_sync_gram
docker compose up -d --build # build container
docker exec -it $CONTAINER_NAME bash # start

poetry install # build python venv and install packages
poetry run bash install_apex.sh
```

### IWSLT'14 German to English
Prepare the dataset
```sh
DATA_DIR=databin/iwslt14_deen
TEXT=data/iwslt14_deen

bash preparation/prepare_iwslt14_deen.sh

poetry run fairseq-preprocess \
    --source-lang de \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir $DATA_DIR \
    --workers 16
```

Train the model
```sh
DATA_DIR=databin/iwslt14_deen
MODEL_DIR=checkpoints/iwslt14_deen_bpe/structformer_rank
TOTAL_UPDATES=15000
SYNC_MEHTOD="rank" # or "mse"

poetry run fairseq-train $DATA_DIR \
    --user-dir src \
    --arch structformer_iwslt_de_en --share-decoder-input-output-embed \
    --induction-layer 1 --alignment-heads 4 --alignment-heads 4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_sync --label-smoothing 0.1 \
    --sync-lambda 0.005 --sync-method $SYNC_MEHTOD --hinge-margen 0.1 \
    --max-tokens 4096 --update-freq $UPDATE_FREQ \
    --no-epoch-checkpoints --save-dir $MODEL_DIR --max-update $TOTAL_UPDATES
```

Evaluate the model
```sh
DATA_DIR=databin/iwslt14_deen
MODEL_DIR=checkpoints/iwslt14_deen_bpe/structformer_rank
RESULT_DIR=logs/iwslt14_deen/bpe/structformer_rank
EVALB=$HOME/thirdparty/benepar/EVALB_SPMRL/evalb_spmrl
EVALB_PARAM=$HOME/thirdparty/benepar/EVALB_SPMRL/spmrl_nolabel.prm
mkdir -p $RESULT_DIR

poetry run fairseq-generate $DATA_DIR \
    --user-dir src --gen-subset test \
    --user-dir src --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 5 --quiet

poetry run python generate_grammar.py $DATA_DIR \
    --gen-subset test \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 64 --beam 1 \
    --results-path $RESULT_DIR

# If you want to do tree evaluation, you have to prepare a gold tree file. In this case, gold file is generated by stanza which stanford parser.
STANFORD_GOLD=experiment/iwslt_deen/nobpe_outputs/structformer/gold.tree # english tree

grep ^TT- $RESULT_DIR/generate-test.txt | sort -V | cut -f2 > $RESULT_DIR/pred.orig.tree
# postprocess to fill in the <unk>
poetry run python complete_tree.py -t data/iwslt14_deen/tmp/test.en < $RESULT_DIR/pred.orig.tree > $RESULT_DIR/pred.cplt.tree

paste $RESULT_DIR/pred.cplt.tree $STANFORD_GOLD | grep -v "|||" > $RESULT_DIR/tmp.tsv
cut -f1 $RESULT_DIR/tmp.tsv > $RESULT_DIR/pred.tree
cut -f2 $RESULT_DIR/tmp.tsv > $RESULT_DIR/gold.tree

$EVALB -p $EVALB_PARAM $RESULT_DIR/gold.tree $RESULT_DIR/pred.tree
```

## Citation
```bibtex
@inproceedings{harada2021neural,
  title={Neural Machine Translation with Synchronous Latent Phrase Structure},
  author={
    Shintaro Harada and
    Taro Watanabe
  },
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Student Research Workshop}
  url={https://aclanthology.org/2021.acl-srw.33/},
  year={2021}
}
```
