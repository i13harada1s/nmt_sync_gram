#!/bin/bash

# following https://arxiv.org/pdf/1909.02074.pdf

########################
#TODO: please set your environments
python="pipenv run python"
THIRDPARTY=$HOME/thirdparty
ALIGN_SCRIPTS=$THIRDPARTY/alignment-scripts
MOSES_SCRIPTS=$THIRDPARTY/mosesdecoder/scripts
FASEQ_SCRIPTS=$THIRDPARTY/fairseq/scripts
DATA_DIR=$ALIGN_SCRIPTS
########################

TOKENIZER=$MOSES_SCRIPTS/tokenizer/tokenizer.perl
LOWERCASE=$MOSES_SCRIPTS/tokenizer/lowercase.perl
CLEAN=$MOSES_SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl

SPM_TRAIN=$FASEQ_SCRIPTS/spm_train.py
SPM_ENCODE=$FASEQ_SCRIPTS/spm_encode.py
# BPE_TOKENS=10000
BPE_TOKENS=32000 # https://aclanthology.org/D19-1453.pdf

function log() { 
    echo "[$( date '+%Y-%m-%d %H:%M:%S' ) INFO]: $1" 
}

src=de
tgt=en
prep=$(pwd)/data/europarl_deen_spm32k
tmp=$prep/tmp
orig=$prep/orig
mkdir -p $prep $orig $tmp

#following https://arxiv.org/pdf/1901.11359.pdf

log "Preparing dataset."
for name in train test; do
    ln -s $DATA_DIR/${name}/deen.lc.src $orig/$name.$src
    ln -s $DATA_DIR/${name}/deen.lc.tgt $orig/$name.$tgt
done
ln -s $DATA_DIR/test/deen.talp $orig/deen.talp
log "Done."

log "Spliting dataset."
for lang in $src $tgt; do
    (head -n -1000 > $tmp/train.$lang; cat > $tmp/valid.$lang) < $orig/train.$lang
    mv $orig/test.$lang $tmp/test.$lang
done
log "Done."

log "Cleaning dataset."
for name in train valid; do
    perl -C $CLEAN $tmp/$name $src $tgt $tmp/$name.clean 1 1000
done

for lang in $src $tgt; do
    mv $tmp/train.clean.$lang $tmp/train.$lang
    mv $tmp/valid.clean.$lang $tmp/valid.$lang
done

################################################################################
# log "Tokenzing train and valid sentences."
# for lang in $src $tgt; do
#     cat $orig/train.$lang | \
#         perl -C $NORM_PUNC $lang | \
#         perl -C $REM_NON_PRINT_CHAR | \
#         perl -C $TOKENIZER -threads 8 -a -l $lang | \
#         perl -C $LOWERCASE > $tmp/tmp.$lang 
#     (head -n -1000 > $tmp/train.tok.$lang; cat > $tmp/valid.tok.$lang) < $tmp/tmp.$lang
#     rm $tmp/tmp.$lang 
# done
# log "Done."

# log "Tokenzing test sentences."
# for lang in $src $tgt; do
#     cat $orig/test.$lang | \
#         perl -C $TOKENIZER -threads 8 -l $lang | \
#         perl -C $LOWERCASE > $tmp/test.tok.$lang
# done
# log "Done."

# log "Cleaning train and valid sentences."
# perl -C $CLEAN $tmp/train.tok $src $tgt $tmp/train.tok.clean 1 1000
# perl -C $CLEAN $tmp/valid.tok $src $tgt $tmp/valid.tok.clean 1 1000
# log "Done."

# for lang in $src $tgt; do
#     mv $tmp/train.tok.clean.$lang $tmp/train.$lang
#     mv $tmp/valid.tok.clean.$lang $tmp/valid.$lang
#     mv $tmp/test.tok.$lang $tmp/test.$lang
# done
################################################################################

echo "" > $tmp/train.deen
for lang in $src $tgt; do
    cat $tmp/train.$lang >> $tmp/train.deen
done

log "Building BPE model on ${tmp}/train.deen"
$python $SPM_TRAIN \
    --input=$tmp/train.deen \
    --model_prefix=$prep/sentencepiece.bpe \
    --vocab_size=$BPE_TOKENS \
    --character_coverage=1.0 \
    --model_type=bpe
log "Done."

for split in train valid test; do
    log "Appling BPE to $tmp/$split.$src and $tmp/$split.$tgt."
    $python $SPM_ENCODE \
        --model $prep/sentencepiece.bpe.model \
        --output_format=piece \
        --inputs $tmp/$split.$src $tmp/$split.$tgt \
        --outputs $prep/$split.$src $prep/$split.$tgt
done

#following https://www.aclweb.org/anthology/2020.emnlp-main.42.pdf
#train 1.9M
#valid 994
#test  508
log "The statistics of preprocessed data."
for lang in $src $tgt; do
    for name in train valid test; do
        echo "$( wc -l  ${prep}/${name}.${lang} )"
    done
done

# 1905082 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/train.de
# 994 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/valid.de
# 508 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/test.de
# 1905082 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/train.en
# 994 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/valid.en
# 508 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/test.en

# 1905698 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/train.de
# 994 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/valid.de
# 508 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/test.de
# 1905698 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/train.en
# 994 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/valid.en
# 508 /home/cl/shintaro-h/fairseq_project/language_modeling/data/europarl_deen/test.en
