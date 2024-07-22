#!/bin/bash

# http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2018/baseline/dataPreparationJE.html

### your environmnet ###
TOKENIZER_JP="juman"
TOKENIZER_ZJ=$HOME/thirdparty/stanford-segmenter-2014-01-04/segment.sh
bpe=$HOME/thirdparty/subword-nmt/subword_nmt
DATA_DIR=$HOME/data/ASPEC/ASPEC-JC
########################

function log() { 
    echo "[$( date '+%Y-%m-%d %H:%M:%S' ) INFO]: $1" 
}

prep=$(pwd)/data/wat18_jazh
tmp=${prep}/tmp
orig=${prep}/orig
mkdir -p ${prep} ${orig} ${tmp}

log "Extracting sentences."
for name in train dev test; do 
    perl -C -ne 'chomp; @a=split/ \|\|\| /; print $a[1], "\n";' < $DATA_DIR/${name}/${name}.txt > ${orig}/${name}.ja.txt 
    perl -C -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < $DATA_DIR/${name}/${name}.txt > ${orig}/${name}.zh.txt
done 

log "Tokenizing sentences in Japanese."
for name in train dev test; do 
    cat ${orig}/${name}.ja.txt | \
        perl -C -pe 'use utf8; s/　/ /g;' | \
        $TOKENIZER_JP -b | \
        perl -C -ne 'chomp; if($_ eq "EOS"){print join(" ",@b),"\n"; @b=();} else {@a=split/ /; push @b, $a[0];}' | \
        perl -C -pe 'use utf8; s/^ +//; s/ +$//; s/ +/ /g;' | \
        perl -C -pe 'use utf8; tr/\|[]/｜［］/; ' \
        > ${tmp}/${name}.ja
done 

log "Tokenizing sentences in Chinese."
for name in train dev test; do 
    $TOKENIZER_ZJ ctb ${orig}/${name}.zh.txt UTF-8 0 | \
    perl -C -pe 'use utf8; tr/\|[]/｜［］/; ' \
    > ${tmp}/${name}.zh
done

log "Building a BPE model."
python ${bpe}/learn_joint_bpe_and_vocab.py \
    --input ${tmp}/train.ja ${tmp}/train.zh -s 100000 -o ${prep}/bpe_codes \
    --write-vocabulary ${prep}/vocab.ja ${prep}/vocab.zh

log "Applying the BPE model."
for name in train dev test; do 
    python ${bpe}/apply_bpe.py -c ${prep}/bpe_codes \
        --vocabulary ${prep}/vocab.ja \
        --vocabulary-threshold 10 \
        < ${tmp}/${name}.ja > ${prep}/${name}.ja
    python ${bpe}/apply_bpe.py -c ${prep}/bpe_codes \
        --vocabulary ${prep}/vocab.zh \
        --vocabulary-threshold 10 \
        < ${tmp}/${name}.zh > ${prep}/${name}.zh
done

log "The statistics of preprocessed data."
for lang in ja zh; do
    for name in train dev test; do
        echo "$( wc -l  ${prep}/${name}.${lang} )"
    done
done
