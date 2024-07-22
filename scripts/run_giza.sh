#!/bin/bash
set -e

# MGIZA MANUAL
# https://hovinh.github.io/blog/2016-04-29-install-mgiza-ubuntu/

ALIGN_SCRIPTS=$HOME/thirdparty/alignments-scripts/scripts
MGIZA_DIR=$HOME/thirdparty/mgiza

# check if MGIZA_DIR is set and installed
if [ -z ${MGIZA_DIR} ]; then
  echo "Please set the variable MGIZA_DIR."
  exit 1
fi

if [ ! -f ${MGIZA_DIR}/mgizapp/bin/mgiza ]; then
  echo "Please install mgiza, file ${MGIZA_DIR}/mgizapp/bin/mgiza not found."
  exit 1
fi

# check parameter count and write usage instruction
if (( $# != 3 )); then
  echo "Usage: $0 source_file_path target_file_path experiment_dir_path"
  exit 1
fi

src_path=$(realpath $1)
tgt_path=$(realpath $2)
src_file=${1##*/} #filename
tgt_file=${2##*/} #filename
src=${1##*.} #extension
tgt=${2##*.} #extension
expdir_path=$3

pushd $expdir_path

# creates vocab and sentence files
${MGIZA_DIR}/mgizapp/bin/plain2snt ${src_path} ${tgt_path}

${MGIZA_DIR}/mgizapp/bin/mkcls -n10 -p${src_path} -V${src_file}.class &
${MGIZA_DIR}/mgizapp/bin/mkcls -n10 -p${tgt_path} -V${tgt_file}.class &
wait

${MGIZA_DIR}/mgizapp/bin/snt2cooc ${src_file}_${tgt_file}.cooc ${src_path}.vcb ${tgt_path}.vcb ${src_path}_${tgt_file}.snt &
${MGIZA_DIR}/mgizapp/bin/snt2cooc ${tgt_file}_${src_file}.cooc ${tgt_path}.vcb ${src_path}.vcb ${tgt_path}_${src_file}.snt &
wait

mkdir -p forward 
config=forward/config.txt
echo "corpusfile ${src_path}_${tgt_file}.snt" > $config
echo "sourcevocabularyfile ${src_path}.vcb" >> $config
echo "targetvocabularyfile ${tgt_path}.vcb" >> $config
echo "coocurrencefile ${src_file}_${tgt_file}.cooc" >> $config
echo "sourcevocabularyclasses ${src_file}.class" >> $config
echo "targetvocabularyclasses ${tgt_file}.class" >> $config

mkdir -p backward
config=backward/config.txt
echo "corpusfile ${tgt_path}_${src_file}.snt" > $config
echo "sourcevocabularyfile ${tgt_path}.vcb" >> $config
echo "targetvocabularyfile ${src_path}.vcb" >> $config
echo "coocurrencefile ${tgt_file}_${src_file}.cooc" >> $config
echo "sourcevocabularyclasses ${tgt_file}.class" >> $config
echo "targetvocabularyclasses ${src_file}.class" >> $config

for name in forward backward; do
  cd $name
  echo "nodumps 0" >> config.txt
  echo "ncpus 8" >> config.txt
  echo "onlyaldumps 1" >> config.txt
  echo "hmmdumpfrequency 5" >> config.txt
  # Run Giza
  echo "Run MGIZA++ in ${name}.."
  ${MGIZA_DIR}/mgizapp/bin/mgiza config.txt
  cat *A3.final.part* > allA3.txt
  cd ..
done

# convert alignments
pipenv run ${ALIGN_SCRIPTS}/a3ToTalp.py < forward/allA3.txt > ${src}_${tgt}.talp
pipenv run ${ALIGN_SCRIPTS}/a3ToTalp.py < backward/allA3.txt > ${tgt}_${src}.talp

popd