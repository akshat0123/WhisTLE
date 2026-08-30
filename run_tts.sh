#!/bin/bash

################################################################################
#                                 Environment
################################################################################
OOD_PATH=<DIRECTORY NAME>
CV_OUT_PATH=<DIRECTORY NAME>
LS_OUT_PATH=<DIRECTORY NAME>

################################################################################
#                    Text-To-Speech Generation: CommonVoice
###############################################################################

conda run --no-capture-output -n fastspeech python -m scripts.gen_tts_fastspeech_cv \
    --inpath $OOD_PATH \
    --outpath $CV_OUT_PATH \
    --limit 10

################################################################################
#                    Text-To-Speech Generation: LibriSpeech
###############################################################################

conda run --no-capture-output -n speecht5 python -m scripts.gen_tts_speecht5_ls \
    --inpath $OOD_PATH \
    --outpath $LS_OUT_PATH \
    --limit 10
