# WhisTLE
Official repository for [WhisTLE: Deeply Supervised, Text-Only Domain Adaptation for Pretrained Speech Recognition Transformers](https://arxiv.org/abs/2509.10452)

## Steps
* Download data: the `datasets` directory contains metadata.jsonl files we used for all "in-domain" and "out-of-domain" datasets in order to have a common metadata schema during all training/testing. Links to each dataset can be found in the "Datasets" section below. Once downloaded, the metadata files can be placed in the root folder of each dataset. All audio was converted into a 16khz wav file format.
* Build environments: the `envs` directory contains conda environment yaml files, with the main `env.yaml` being for all training/testing, and the remaining two environments being for TTS audio generation.
* Generate TTS data: For [Fastspeech2](https://huggingface.co/facebook/fastspeech2-en-ljspeech) tts audio generation, create the `fastspeech` environment provided in the `envs` folder. For [SpeechT5](https://huggingface.co/microsoft/speecht5_tts) tts audio generation, create the `speecht5` environment provided in the `envs` folder. Once both environments are created, you can run the `run_tts.sh` script to generate TTS audios for training, or simply use the script as an example use case. After creating both conda environments, set the environment variables at the top of the `run_tts.sh` script and run it.
* Run training/evaluation scripts: To run all training/evaluation create the `tle` environment provided in the `envs` folder. The `run_experiments.sh` script demonstrates an example of how to run Whisper-large experiments using CommonVoice as the "in-domain" dataset and EMNS as the "out-of-domain" dataset.

## Datasets
Below are the sources for all datasets used in the work:
* [CommonVoice](https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu)
* [LibriSpeech](https://www.openslr.org/12)
* [EMNS](https://www.openslr.org/136/)
* [EmoV\_DB](https://www.openslr.org/115/)
* [ST-AEDS](https://www.openslr.org/45/)
* [EABI](https://www.openslr.org/83/)
