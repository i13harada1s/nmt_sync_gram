#!/bin/bash
set -e

export PATH=$HOME/.linuxbrew/Cellar/boost/1.75.0_3/:$PATH
export LD_LIBRARY_PATH=$HOME/.linuxbrew/Cellar/boost/1.75.0_3/lib:$LD_LIBRARY_PATH
export INCLUDE=$HOME/.linuxbrew/Cellar/boost/1.75.0_3/include:$INCLUDE

ALIGN_SCRIPTS=$HOME/thirdparty/alignment-scripts
REFERENCE=$ALIGN_SCRIPTS/test/enfr.talp

src=en
tgt=fr

log() {
    echo "$( date '+%Y-%m-%d %H:%M:%S' ) INFO: $1"
}

check_file() {
    if [ ! -e $1 ]; then
        echo "$1 does not exist."
        exit 1
    fi
}

n_reference=`cat $REFERENCE | wc -l`

#WORD
DATADIR=data/europarl_enfr_spm32k/tmp
EXPERIMENT=experiments/results/europarl_enfr_spm32k/giza
mkdir -p $EXPERIMENT

log "Create train data for run_giza.sh."
for l in $src $tgt; do
    check_file $DATADIR/train.$l
    check_file $DATADIR/valid.$l
    check_file $DATADIR/test.$l
    cat $DATADIR/train.$l $DATADIR/valid.$l $DATADIR/test.$l > $EXPERIMENT/train.align.plustest.$l
done

log "Runnig giza..."
./scripts/run_giza.sh $EXPERIMENT/train.align.plustest.$src $EXPERIMENT/train.align.plustest.$tgt $EXPERIMENT
log "Done."

# get aliginment of test set
tail -n $n_reference $EXPERIMENT/${src}_${tgt}.talp > $EXPERIMENT/${src}_${tgt}.test.talp 
tail -n $n_reference $EXPERIMENT/${tgt}_${src}.talp > $EXPERIMENT/${tgt}_${src}.test.talp 

log "Evaluate the alignments."
method="grow-diagonal"
pipenv run python $ALIGN_SCRIPTS/scripts/combine_bidirectional_alignments.py \
    $EXPERIMENT/${src}_${tgt}.test.talp $EXPERIMENT/${tgt}_${src}.test.talp --method $method \
    > $EXPERIMENT/${src}_${tgt}.bidir.${method}.talp

log "${src}-${tgt}"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${src}_${tgt}.talp --oneRef --fAlpha 0.5
log "${tgt}-${tgt}"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${tgt}_${src}.talp --oneRef --fAlpha 0.5 --reverseRef
log "${src}-${tgt} bidirectional"
$ALIGN_SCRIPTS/scripts/aer.py $REFERENCE $EXPERIMENT/${src}_${tgt}.bidir.${method}.talp --oneRef --fAlpha 0.5