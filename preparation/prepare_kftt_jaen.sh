#!/bin/bash

# TRAIN=data/kftt-data-1.0/data/orig/kyoto-train.ja # 440288
# VALID=data/kftt-data-1.0/data/orig/kyoto-dev.ja # 1166
# TEST=data/kftt-alignments/data/japanese.txt # 1235
# ALIGN=/home/cl/shintaro-h/data/kftt-alignments/data/align.txt # 1235

########################
#TODO: please set your environments
python="pipenv run python"
THIRDPARTY=$HOME/thirdparty
KYTEA_TOKENIZER="kytea"
PARALLEL="parallel --no-notice --pipe -j 16 -k"
MOSES_SCRIPTS=$THIRDPARTY/mosesdecoder/scripts
FASEQ_SCRIPTS=$THIRDPARTY/fairseq/scripts
NMT_DATA_DIR=$HOME/data/kftt-data-1.0/data/tok
ALINMENT_DIR=$HOME/data/kftt-alignments/data
########################

MOSES_TOKENIZER=$MOSES_SCRIPTS/tokenizer/tokenizer.perl
LOWERCASE=$MOSES_SCRIPTS/tokenizer/lowercase.perl
CLEAN=$MOSES_SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl

SPM_TRAIN=$FASEQ_SCRIPTS/spm_train.py
SPM_ENCODE=$FASEQ_SCRIPTS/spm_encode.py

# BPE_TOKENS=16000
BPE_TOKENS=32000 # https://aclanthology.org/D19-1453.pdf

function log() { 
    echo "[$( date '+%Y-%m-%d %H:%M:%S' ) INFO]: $1" 
}

src=ja
tgt=en
prep=$(pwd)/data/kftt_jaen_spm32k
tmp=$prep/tmp
orig=$prep/orig
mkdir -p $prep $orig $tmp

log "Fetch courpus." # following https://aclanthology.org/D16-1210.pdf
ln -s $NMT_DATA_DIR/kyoto-train.cln.ja $tmp/train.ja # move to tmpdir because do not lowercase
ln -s $NMT_DATA_DIR/kyoto-train.cln.en $orig/train.en

cat $ALINMENT_DIR/japanese-00[1-8].txt > $tmp/dev.ja
cat $ALINMENT_DIR/japanese-009.txt $ALINMENT_DIR/japanese-01[0-5].txt > $tmp/test.ja

cat $ALINMENT_DIR/english-00[1-8].txt > $tmp/dev.en
cat $ALINMENT_DIR/english-009.txt $ALINMENT_DIR/english-01[0-5].txt > $tmp/test.en

log "Lowercase sentences in English"
cat $orig/train.en | perl -C $LOWERCASE > $tmp/train.en

echo "" > $tmp/train.jaen
for lang in $src $tgt; do
    cat $tmp/train.$lang >> $tmp/train.jaen
done

log "Building BPE model on ${tmp}/train.jaen"
$python $SPM_TRAIN \
    --input=$tmp/train.jaen \
    --model_prefix=$prep/sentencepiece.bpe \
    --vocab_size=$BPE_TOKENS \
    --character_coverage=0.9995 \
    --model_type=bpe
log "Done."

for split in train dev test; do
    log "Appling BPE to $tmp/$split.$src and $tmp/$split.$tgt."
    $python $SPM_ENCODE \
        --model $prep/sentencepiece.bpe.model \
        --output_format=piece \
        --inputs $tmp/$split.$src $tmp/$split.$tgt \
        --outputs $prep/$split.$src $prep/$split.$tgt
done

#following https://aclanthology.org/D16-1210.pdf
#train 330K -> 329882 (333K)
#valid 653  -> 653
#test  582  -> 582
log "The statistics of preprocessed data."
for lang in $src $tgt; do
    for name in train dev test; do
        echo "$( wc -l  ${prep}/${name}.${lang} )"
    done
done
