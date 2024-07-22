#!/bin/bash
set -e

FASTALIGN_DIR=$HOME/thirdparty/fast_align

# check if FASTALIGN_DIR is set and installed
if [ -z ${FASTALIGN_DIR} ]; then
  echo "Please set the variable FASTALIGN_DIR."
  exit 1
fi

if [ ! -f ${FASTALIGN_DIR}/build/fast_align ]; then
  echo "Please install fastalign, file ${FASTALIGN_DIR}/build/fast_align not found."
  exit 1
fi

# check parameter count and write usage instruction
if (( $# != 3 )); then
  echo "Usage: $0 source_file_path target_file_path experiment_dir_path."
  exit 1
fi

src_path=$1
tgt_path=$2
src=${1##*.} #extension
tgt=${2##*.} #extension
expdir_path=$3

# create format used for fastalign
paste -d "~" ${src_path} ${tgt_path} | sed 's/~/ ||| /g' > $expdir_path/${src}_${tgt}
paste -d "~" ${tgt_path} ${src_path} | sed 's/~/ ||| /g' > $expdir_path/${tgt}_${src}

# remove lines which have an empty  src or tgt
sed -e '/^ |||/d' -e '/||| $/d' $expdir_path/${src}_${tgt} > $expdir_path/${src}_${tgt}.clean
sed -e '/^ |||/d' -e '/||| $/d' $expdir_path/${tgt}_${src} > $expdir_path/${tgt}_${src}.clean

# align in both directions
${FASTALIGN_DIR}/build/fast_align -i $expdir_path/${src}_${tgt}.clean -p $expdir_path/${src}_${tgt}.model -d -o -v > $expdir_path/${src}_${tgt}.talp 2> $expdir_path/${src}_${tgt}.error
${FASTALIGN_DIR}/build/fast_align -i $expdir_path/${tgt}_${src}.clean -p $expdir_path/${tgt}_${src}.model -d -o -v > $expdir_path/${tgt}_${src}.talp 2> $expdir_path/${tgt}_${src}.error