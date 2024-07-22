# Dataset Preparation

## Translation Task
###  WMT16 English-Garman
```bash
cd path/to/your/workspace

# reference: https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md

TEXT=path/to/wmt16_en_de
fairseq-preprocess \
    --source-lang en \
    --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_ende_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 16
```

### ASPEC Japanese-English
```bash
cd path/to/your/workspace

./preparation/prepare_wat18_jaen.sh

TEXT=data/wat18_jaen
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --destdir databin/wat18_jaen \
    --workers 16
```

### ASPEC Japanese-Chinese
```bash
cd path/to/your/workspace

./preparation/prepare_wat18_jazh.sh

TEXT=data/wat18_jazh
fairseq-preprocess \
    --source-lang ja \
    --target-lang zh \
    --trainpref $TEXT/train \
    --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --destdir databin/wat18_jazh \
    --workers 16
```

## Parsing Task
###  IWSLT14 Garman-English
```bash
cd path/to/your/workspace

./preparation/prepare_iwslt14_deen.sh

# BPE-wise
TEXT=data/iwslt14_deen
fairseq-preprocess \
    --source-lang de \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir databin/iwslt14_deen \
    --workers 16

# word-wise
TEXT=data/iwslt14_deen/tmp
fairseq-preprocess \
    --source-lang de \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir databin/iwslt14_deen_nobpe \
    --workers 16
```

```bash
TEXT=data/iwslt14_deen

#benepar
# pip install -U pip setuptools wheel
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python benepar_parse.py \
#     --language en \
#     --input $TEXT/tmp/test.en \
#     --output $TEXT/benepar/test.en.tree \
#     --remove-tags --remove-root
```

#stanford
```bash
pip install -U stanza
```

```python
CORENLP_HOME=/path/to/your/corenlp
stanza.install_corenlp(dir=CORENLP_HOME)
stanza.download_corenlp_models(model='english', version='4.1.0', dir=CORENLP_HOME)
# stanza.download_corenlp_models(model='arabic', version='4.1.0', dir=CORENLP_HOME)
# stanza.download_corenlp_models(model='chinese', version='4.1.0', dir=CORENLP_HOME)
# stanza.download_corenlp_models(model='english-kbp', version='4.1.0', dir=CORENLP_HOME)
# stanza.download_corenlp_models(model='french', version='4.1.0', dir=CORENLP_HOME)
# stanza.download_corenlp_models(model='german', version='4.1.0', dir=CORENLP_HOME)
# stanza.download_corenlp_models(model='spanish', version='4.1.0', dir=CORENLP_HOME)
```

```bash
TEXT=data/iwslt14_deen
export CORENLP_HOME=/path/to/your/corenlp
python stanford_parse.py \
    -l "english" \
    -p scripts/properties/english.properties
    -i $TEXT/tmp/test.en \
    -o $TEXT/stanford/test.en.tree \
    --workers 16
paset text.1 text.2 > text.merge
grep -v "separeter" text.merge > text.rm.merge
cut -f1 text.rm.merge > text.rm.tree
```

## Alignment Task
### German-English
```bash
cd path/to/your/workspace

./preparation/prepare_align_deen.sh

TEXT=data/align_deen_spm32k
fairseq-preprocess \
    --source-lang de \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir databin/align_deen_spm32k \
    --joined-dictionary \
    --workers 16
```

### French-English
```bash
cd path/to/your/workspace

./preparation/prepare_align_enfr.sh

TEXT=data/align_enfr_spm32k
fairseq-preprocess \
    --source-lang en \
    --target-lang fr \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir databin/align_enfr_spm32k \
    --joined-dictionary \
    --workers 16
```

### Rossian-English
```bash
cd path/to/your/workspace

./preparation/prepare_align_roen.sh

TEXT=data/align_roen_spm32k
fairseq-preprocess \
    --source-lang ro \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir databin/align_roen_spm32k \
    --joined-dictionary \
    --workers 16
```

### KFTT Japanese-English
```bash
cd path/to/your/workspace

./preparation/prepare_kftt_jaen.sh

TEXT=data/kftt_jaen_spm32k
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --destdir databin/kftt_jaen_spm32k \
    --joined-dictionary \
    --workers 16
```
