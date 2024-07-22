#!/bin/bash
set -eu

FASTALIGN_DIR=$HOME/thirdparty/fast_align
ALIGN_SCRIPTS=$HOME/thirdparty/alignment-scripts
REFERENCE=$ALIGN_SCRIPTS/test/enfr.talp

src=en
tgt=fr

info() {
    echo "$( date '+%Y-%m-%d %H:%M:%S' ) INFO: $1"
}

error() {
    echo "$( date '+%Y-%m-%d %H:%M:%S' ) ERROR: $1"
    exit 1
}

check_file() {
    if [ ! -e $1 ]; then
        error "$1 does not exist."
    fi
}

n_reference=`cat $REFERENCE | wc -l`

# WORD
info "do experiments on word-level"
DATADIR=data/europarl_enfr_spm32k/tmp
EXPERIMENT=experiments/results/europarl_enfr_spm32k/fastalign_tkn
mkdir -p $EXPERIMENT
info "create train data for run_fastalign.sh."
for l in $src $tgt; do
    check_file $DATADIR/train.$l
    check_file $DATADIR/valid.$l
    check_file $DATADIR/test.$l
    cat $DATADIR/train.$l $DATADIR/valid.$l $DATADIR/test.$l > $EXPERIMENT/train.align.plustest.$l
done
info "done."

info "running fastalign..."
./scripts/run_fastalign.sh $EXPERIMENT/train.align.plustest.$src $EXPERIMENT/train.align.plustest.$tgt $EXPERIMENT
info "done."

# get aliginment of test set
tail -n $n_reference $EXPERIMENT/${src}_${tgt}.talp > $EXPERIMENT/${src}_${tgt}.test.talp 
tail -n $n_reference $EXPERIMENT/${tgt}_${src}.talp > $EXPERIMENT/${tgt}_${src}.test.talp 

info "evaluate the alignments."
method="grow-diagonal"
pipenv run python $ALIGN_SCRIPTS/scripts/combine_bidirectional_alignments.py $EXPERIMENT/${src}_${tgt}.test.talp $EXPERIMENT/${tgt}_${src}.test.talp --method $method > $EXPERIMENT/${src}_${tgt}.test.bidir.${method}.talp

info "${src}-${tgt}"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${src}_${tgt}.test.talp --oneRef --fAlpha 0.5
info "${tgt}-${tgt}"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${tgt}_${src}.test.talp --oneRef --fAlpha 0.5 --reverseRef
info "${src}-${tgt} bidirectional"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${src}_${tgt}.test.bidir.${method}.talp --oneRef --fAlpha 0.5
info "done."

echo ""

# # BPE(SentencePiece)
info "do experiments on sentencepiece-level"
DATADIR=data/europarl_enfr_spm32k
EXPERIMENT=experiments/results/europarl_enfr_spm32k/fastalign_spm
mkdir -p $EXPERIMENT
info "create train data for run_fastalign.sh."
for l in $src $tgt; do
    check_file $DATADIR/train.$l
    check_file $DATADIR/valid.$l
    check_file $DATADIR/test.$l
    cat $DATADIR/train.$l $DATADIR/valid.$l $DATADIR/test.$l > $EXPERIMENT/train.align.plustest.$l
done
info "done."

info "running fastalign..."
./scripts/run_fastalign.sh $EXPERIMENT/train.align.plustest.$src $EXPERIMENT/train.align.plustest.$tgt $EXPERIMENT
info "done."

# get aliginment of test set
tail -n $n_reference $EXPERIMENT/${src}_${tgt}.talp > $EXPERIMENT/${src}_${tgt}.test.bpe.talp 
tail -n $n_reference $EXPERIMENT/${tgt}_${src}.talp > $EXPERIMENT/${tgt}_${src}.test.bpe.talp 

pipenv run python $ALIGN_SCRIPTS/scripts/sentencepiece_to_word_alignments.py \
    $DATADIR/test.${src} $DATADIR/test.${tgt} < $EXPERIMENT/${src}_${tgt}.test.bpe.talp  > $EXPERIMENT/${src}_${tgt}.test.talp
pipenv run python $ALIGN_SCRIPTS/scripts/sentencepiece_to_word_alignments.py \
    $DATADIR/test.${tgt} $DATADIR/test.${src} < $EXPERIMENT/${tgt}_${src}.test.bpe.talp  > $EXPERIMENT/${tgt}_${src}.test.talp

info "evaluate the alignments."
method="grow-diagonal"
pipenv run python $ALIGN_SCRIPTS/scripts/combine_bidirectional_alignments.py $EXPERIMENT/${src}_${tgt}.test.talp $EXPERIMENT/${tgt}_${src}.test.talp --method $method > $EXPERIMENT/${src}_${tgt}.test.bidir.${method}.talp

info "${src}-${tgt}"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${src}_${tgt}.test.talp --oneRef --fAlpha 0.5
info "${tgt}-${tgt}"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${tgt}_${src}.test.talp --oneRef --fAlpha 0.5 --reverseRef
info "${src}-${tgt} bidirectional"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${src}_${tgt}.test.bidir.${method}.talp --oneRef --fAlpha 0.5
info "done."