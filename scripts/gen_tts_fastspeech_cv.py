
from argparse import ArgumentParser
import json, os

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

from torchaudio.functional import resample
from tqdm import trange, tqdm
import soundfile as sf

def main():

    parser = ArgumentParser()
    parser.add_argument('-i', '--inpath', type=str, required=True)
    parser.add_argument('-o', '--outpath', type=str, required=True)
    parser.add_argument('-l', '--limit', type=int, default=None)
    args = parser.parse_args()
    inpath = args.inpath
    outpath = args.outpath
    limit = args.limit

    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-200_speaker-cv4",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )

    models[0] = models[0].cuda()
    model = models[0]

    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(models, cfg)

    for dirname in sorted(os.listdir(inpath)):
        dirpath = f'{inpath}/{dirname}'

        if dirname not in os.listdir(outpath):
            os.mkdir(f"{outpath}/{dirname}")
            os.mkdir(f"{outpath}/{dirname}/data")

        data = [json.loads(x) for x in open(f'{dirpath}/metadata.jsonl')]

        if limit is not None:
            data = data[:limit]

        metadata = []

        for item in tqdm(data, ncols=80):
            text = item['transcription']
            writepath = f"{outpath}/{dirname}/data/{item['path'].split('/')[-1]}"
            metapath = f"data/{item['path'].split('/')[-1]}"

            try:
                sample = TTSHubInterface.get_model_input(task, text)
                sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].cuda()
                sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].cuda()
                sample["speaker"] = sample["speaker"].cuda()
                wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
                wav = resample(wav, rate, 16000)

                sf.write(writepath, wav.cpu().numpy(), samplerate=16000)
                metadata.append({ 'transcription': text, 'path': metapath })
            
            except:
                continue

        with open(f"{outpath}/{dirname}/metadata.jsonl", 'w', encoding='utf-8') as outfile:
            for item in metadata:
                outfile.write(f"{json.dumps(item)}\n")
    

if __name__ == '__main__':
    main()
