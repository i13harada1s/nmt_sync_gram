#!/bin/bash

# http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2018/baseline/dataPreparationJE.html

### your environmnet ###
juman="juman"
bpe="subword-nmt"
THIRDPARTY=$HOME/thirdparty
MOSES_SCRIPTS=$THIRDPARTY/mosesdecoder/scripts
WAT18_SCRIPTS=$THIRDPARTY/script.converter.distribution
DATA_DIR=$HOME/data/ASPEC/ASPEC-JE
########################

function log() { 
    echo "[$( date '+%Y-%m-%d %H:%M:%S' ) INFO]: $1" 
}

prep=$(pwd)/data/wat18_jaen
tmp=${prep}/tmp
orig=${prep}/orig
mkdir -p ${prep} ${orig} ${tmp}


log "Extracting sentences."
for name in dev test; do 
    perl -C -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < $DATA_DIR/${name}/${name}.txt > ${orig}/${name}.ja.txt 
    perl -C -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < $DATA_DIR/${name}/${name}.txt > ${orig}/${name}.en.txt 
done 
for name in train-1 train-2 train-3; do 
    perl -C -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < $DATA_DIR/train/${name}.txt > ${orig}/${name}.ja.txt 
    perl -C -ne 'chomp; @a=split/ \|\|\| /; print $a[4], "\n";' < $DATA_DIR/train/${name}.txt > ${orig}/${name}.en.txt 
done 

log "Removing date expressions at EOS in Japanese in the training and development data to reduce noise."
for name in train-1 train-2 train-3 dev; do 
    mv ${orig}/${name}.ja.txt ${orig}/${name}.ja.txt.org 
    cat ${orig}/${name}.ja.txt.org | perl -C -pe 'use utf8; s/(.)［[０-９．]+］$/${1}/;' > ${orig}/${name}.ja.txt 
    rm ${orig}/${name}.ja.txt.org # remove tmp file
done 

log "Tokenizing sentences in Japanese."
for name in train-1 dev test; do 
    cat ${orig}/${name}.ja.txt | \
        perl -C -pe 'use utf8; s/　/ /g;' | \
        ${juman} -b | \
        perl -C -ne 'use utf8; chomp; if($_ eq "EOS"){print join(" ",@b),"\n"; @b=();} else {@a=split/ /; push @b, $a[0];}' | \
        perl -C -pe 'use utf8; s/^ +//; s/ +$//; s/ +/ /g;' | \
        perl -C -pe 'use utf8; tr/\|[]/｜［］/; ' \
        > ${tmp}/${name}.ja
done 

log "Tokenizing sentences in English."
for name in train-1 dev test; do 
    cat ${orig}/${name}.en.txt | \
        perl -C ${WAT18_SCRIPTS}/z2h-utf8.pl | \
        perl -C ${MOSES_SCRIPTS}/tokenizer/tokenizer.perl -l en -no-escape \
        > ${tmp}/${name}.en 
done

ln -s ${tmp}/train-1.ja ${tmp}/train.ja
ln -s ${tmp}/train-1.en ${tmp}/train.en

log "Building a BPE model."
${bpe} learn-joint-bpe-and-vocab \
    --input ${tmp}/train.ja ${tmp}/train.en -s 100000 -o ${prep}/bpe_codes \
    --write-vocabulary ${prep}/vocab.ja ${prep}/vocab.en

log "Applying the BPE model."
for name in train dev test; do 
    ${bpe} apply-bpe -c ${prep}/bpe_codes \
        --vocabulary ${prep}/vocab.ja \
        --vocabulary-threshold 10 \
        < ${tmp}/${name}.ja > ${prep}/${name}.ja
    ${bpe} apply-bpe -c ${prep}/bpe_codes \
        --vocabulary ${prep}/vocab.en \
        --vocabulary-threshold 10 \
        < ${tmp}/${name}.en > ${prep}/${name}.en
done

log "The statistics of preprocessed data."
for lang in ja en; do
    for name in train dev test; do
        echo "$( wc -l  ${prep}/${name}.${lang} )"
    done
done
