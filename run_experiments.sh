#!/bin/bash

################################################################################
#                                 Environment
################################################################################
MODEL_TYPE=large
MODEL_PATH=<DIRECTORY NAME>
LOGGING_PATH=<DIRECTORY NAME>
OUTPUT_PATH=<TXT FILE PATH>

CV_PATH=<CommonVoice metadata-training.jsonl FILE PATH>
EMNS_PATH=<EMNS metadata.jsonl FILE PATH>
TTS_EMNS_PATH=<Fastspeech-generated EMNS metadata.jsonl FILE PATH>

################################################################################
#                                  TLE Module
################################################################################

printf "Training TLE Module\n"
python -m scripts.train_tle \
    --model_type $MODEL_TYPE \
    --audio_path $CV_PATH \
    --batch_size 4 \
    --grad_acc 2 \
    --lr 1.25e-5 \
    --name "${LOGGING_PATH}/toa" \
    --out_path "${MODEL_PATH}/toa.pt" \
    --summary \
    --total_steps 100000

################################################################################
#                             No adaptation method
################################################################################

printf "Training Whisper\n"
python3 -m scripts.train_whisper_lm \
    --model_type $MODEL_TYPE \
    --audio_path $CV_PATH \
    --batch_size 4 \
    --grad_acc 2 \
    --lr 1e-4 \
    --name "${LOGGING_PATH}/whisper_lm" \
    --out_path "${MODEL_PATH}/whisper_lm.pt" \
    --summary \
    --total_steps 1000

python3 -m scripts.train \
    --model_type $MODEL_TYPE \
    --audio_path $CV_PATH \
    --batch_size 1 \
    --grad_acc 8 \
    --in_path_whisper "${MODEL_PATH}/whisper_lm.pt" \
    --lr_audio 6.25e-8 \
    --name "${LOGGING_PATH}/model_1" \
    --out_path "${MODEL_PATH}/model_1.pt" \
    --summary \
    --total_steps 100000

printf "Evaluating None\n"
printf "None\n" >> $OUTPUT_PATH
python3 -m scripts.evaluate \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_1.pt" \
    --data_path $EMNS_PATH \
    --batch_size 16 >> $OUTPUT_PATH

################################################################################
#                                     TLE
################################################################################

printf "Training TLE-based model\n"
python3 -m scripts.train \
    --model_type $MODEL_TYPE \
    --audio_multiplier 2 \
    --audio_path $CV_PATH \
    --batch_size 1 \
    --grad_acc 8 \
    --in_path_toa "${MODEL_PATH}/toa.pt" \
    --in_path_whisper "${MODEL_PATH}/whisper_lm.pt" \
    --lr_audio 6.25e-8 \
    --lr_text 6.25e-8 \
    --name "${LOGGING_PATH}/model_2" \
    --out_path "${MODEL_PATH}/model_2.pt" \
    --summary \
    --text_only \
    --text_path $EMNS_PATH \
    --total_steps 100000

printf "Evaluating TLE\n"
printf "Evaluating TLE\n" >> $OUTPUT_PATH
python3 -m scripts.evaluate \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_2.pt" \
    --data_path $EMNS_PATH \
    --limit 10 \
    --batch_size 16 >> $OUTPUT_PATH

################################################################################
#                                     TTS
################################################################################

printf "Training TTS-based model\n"
python3 -m scripts.train \
    --model_type $MODEL_TYPE \
    --audio_path $TTS_EMNS_PATH \
    --batch_size 1 \
    --grad_acc 8 \
    --in_path_whisper "${MODEL_PATH}/model_1.pt" \
    --lr_audio 6.25e-8 \
    --name "${LOGGING_PATH}/model_3" \
    --out_path "${MODEL_PATH}/model_3.pt" \
    --summary \
    --total_steps 50000

printf "Evaluating TTS\n"
printf "Evaluating TTS\n" >> $OUTPUT_PATH 
python3 -m scripts.evaluate \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_3.pt" \
    --data_path $EMNS_PATH \
    --batch_size 16 >> $OUTPUT_PATH 

################################################################################
#                                      SF
################################################################################

printf "Evaluating SF\n"
printf "Evaluating SF\n" >> $OUTPUT_PATH
python3 -m scripts.shallow_fusion \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_1.pt" \
    --text_path $EMNS_PATH \
    --test_path $EMNS_PATH \
    --beta 0.10 \
    --batch_size 4 >> $OUTPUT_PATH 

################################################################################
#                                      DF
################################################################################

printf "Evaluating DF\n" 
printf "Evaluating DF\n" >> $OUTPUT_PATH
python3 -m scripts.deep_fusion \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_1.pt" \
    --train_path $CV_PATH \
    --test_path $EMNS_PATH \
    --text_path $EMNS_PATH \
    --batch_size 4 \
    --total_steps 10000 \
    --lr 1e-4 >> $OUTPUT_PATH

################################################################################
#                                     ILME
################################################################################

printf "Evaluating ILME\n" 
printf "Evaluating ILME\n" >> $OUTPUT_PATH
python3 -m scripts.ilme \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_1.pt" \
    --text_path $EMNS_PATH \
    --test_path $EMNS_PATH \
    --alpha 0.10 \
    --beta 0.10 \
    --batch_size 2 >> $OUTPUT_PATH

################################################################################
#                                   TLE + SF
################################################################################

printf "Evaluating TLE+SF\n"
printf "Evaluating TLE+SF\n" >> $OUTPUT_PATH
python3 -m scripts.shallow_fusion \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_2.pt" \
    --text_path $EMNS_PATH \
    --test_path $EMNS_PATH \
    --beta 0.10 \
    --batch_size 4 >> $OUTPUT_PATH

################################################################################
#                                   TTS + SF
################################################################################

printf "Evaluating TTS+SF\n"
printf "Evaluating TTS+SF\n" >> $OUTPUT_PATH 
python3 -m scripts.shallow_fusion \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_3.pt" \
    --text_path $EMNS_PATH \
    --test_path $EMNS_PATH \
    --beta 0.25 \
    --batch_size 4 >> $OUTPUT_PATH 

#################################################################################
#                                   TLE + DF
################################################################################

printf "Evaluating TLE+DF\n" 
printf "Evaluating TLE+DF\n" >> $OUTPUT_PATH
python3 -m scripts.deep_fusion \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_2.pt" \
    --train_path $CV_PATH \
    --test_path $EMNS_PATH \
    --text_path $EMNS_PATH \
    --batch_size 4 \
    --total_steps 10000 \
    --lr 1e-4 >> $OUTPUT_PATH

################################################################################
#                                   TTS + DF
################################################################################

printf "Evaluating TTS+DF\n" 
printf "Evaluating TTS+DF\n" >> $OUTPUT_PATH
python3 -m scripts.deep_fusion \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_3.pt" \
    --train_path $TTS_EMNS_PATH \
    --test_path $EMNS_PATH \
    --text_path $EMNS_PATH \
    --batch_size 4 \
    --total_steps 10000 \
    --lr 1e-4 >> $OUTPUT_PATH

################################################################################
#                                  TLE + ILME
################################################################################
printf "Evaluating TLE+ILME\n" 
printf "Evaluating TLE+ILME\n" >> $OUTPUT_PATH
python3 -m scripts.ilme \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_2.pt" \
    --text_path $EMNS_PATH \
    --test_path $EMNS_PATH \
    --alpha 0.25 \
    --beta 0.25 \
    --batch_size 2 >> $OUTPUT_PATH

################################################################################
#                                  TTS + ILME
################################################################################

printf "Evaluating TTS+ILME\n"  
printf "Evaluating TTS+ILME\n" >> $OUTPUT_PATH 
python3 -m scripts.ilme \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_3.pt" \
    --text_path $EMNS_PATH \
    --test_path $EMNS_PATH \
    --alpha 0.10 \
    --beta 0.25 \
    --batch_size 2 >> $OUTPUT_PATH

################################################################################
#                                  TLE + TTS
################################################################################

printf "Training TLE+TTS-based model\n"
python2 -m scripts.train \
    --model_type $MODEL_TYPE \
    --audio_multiplier 1 \
    --audio_path $TTS_EMNS_PATH \
    --batch_size 1 \
    --grad_acc 8
    --in_path_toa "${MODEL_PATH}/toa.pt" \
    --in_path_whisper "${MODEL_PATH}/model_1.pt" \
    --lr_audio 6.25e-8 \
    --lr_text 6.25e-8 \
    --name "${LOGGING_PATH}/model_5" \
    --out_path "${MODEL_PATH}/model_5.pt" \
    --summary \
    --text_only \
    --text_path $EMNS_PATH \
    --total_steps 50000

printf "Evaluating TLE+TTS\n"
printf "Evaluating TLE+TTS\n" >> $OUTPUT_PATH 
python3 -m scripts.evaluate \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_5.pt" \
    --data_path $EMNS_PATH \
    --batch_size 16 >> $OUTPUT_PATH 

################################################################################
#                                TLE + TTS + SF
################################################################################

printf "Evaluating TLE+TTS+SF\n"
printf "Evaluating TLE+TTS+SF\n" >> $OUTPUT_PATH 
python3 -m scripts.shallow_fusion \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_5.pt" \
    --text_path $EMNS_PATH \
    --test_path $EMNS_PATH \
    --beta 0.25 \
    --batch_size 4  >> $OUTPUT_PATH 

################################################################################
#                                TLE + TTS + DF
################################################################################

printf "Evaluating TLE+TTS+DF\n" 
printf "Evaluating TLE+TTS+DF\n" >> $OUTPUT_PATH
python3 -m scripts.deep_fusion \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_5.pt" \
    --train_path $TTS_EMNS_PATH \
    --test_path $EMNS_PATH \
    --text_path $EMNS_PATH \
    --batch_size 4 \
    --total_steps 10000 \
    --lr 1e-4 >> $OUTPUT_PATH

################################################################################
#                               TLE + TTS + ILME
################################################################################

printf "Evaluating TLE+TTS+ILME\n"  
printf "Evaluating TLE+TTS+ILME\n" >> $OUTPUT_PATH 
python3 -m scripts.ilme \
    --model_type $MODEL_TYPE \
    --model_path "${MODEL_PATH}/model_5.pt" \
    --text_path $EMNS_PATH \
    --test_path $EMNS_PATH \
    --alpha 0.10 \
    --beta 0.25 \
    --batch_size 2  >> $OUTPUT_PATH

################################################################################
#                                   Baseline
################################################################################

printf "Training OOD-trained model\n"
python3 -m  scripts.train \
    --model_type $MODEL_TYPE \
    --audio_path $EMNS_PATH \
    --batch_size 1 \
    --grad_acc 8 \
    --lr_audio 1e-7 \
    --name "${LOGGING_PATH}/model_7" \
    --out_path "${MODEL_PATH}/model_7.pt" \
    --summary  \
    --total_steps  50000

printf "Evaluating OOD-trained model\n"
printf "Evaluating OOD-trained model\n" >> $OUTPUT_PATH 
python3 -m  scripts.evaluate \
    --model_type  $MODEL_TYPE \
    --model_path  "${MODEL_PATH}/model_7.pt" \
    --data_path  $EMNS_PATH \
    --batch_size  16  >>  $OUTPUT_PATH 
