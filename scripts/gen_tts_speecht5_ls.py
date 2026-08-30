
from argparse import ArgumentParser
from random import randint
import json, os

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from tqdm import tqdm, trange
import soundfile as sf
import torch

def main():

    parser = ArgumentParser()
    parser.add_argument('-i', '--inpath', type=str, required=True)
    parser.add_argument('-o', '--outpath', type=str, required=True)
    parser.add_argument('-l', '--limit', type=int, default=None)
    args = parser.parse_args()
    inpath = args.inpath
    outpath = args.outpath
    limit = args.limit

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").cuda()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").cuda()

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    embeddings = []
    for item in tqdm(embeddings_dataset, ncols=80):
        if '_bdl_' in item['filename'] or \
           '_slt_' in item['filename'] or \
           '_rms_' in item['filename'] or \
           '_clb_' in item['filename']:
               embeddings.append(item)

    for dirname in sorted(os.listdir(inpath)):
        dirpath = f'{inpath}/{dirname}'

        if dirname not in os.listdir(outpath):
            os.mkdir(f"{outpath}/{dirname}")
            os.mkdir(f"{outpath}/{dirname}/data")

        data = [json.loads(x) for x in open(f'{dirpath}/metadata.jsonl')]

        if limit is not None:
            data = data[:limit]

        metadata = []

        for item in tqdm(data, ncols=80, smoothing=0.01):
            text = item['transcription']
            writepath = f"{outpath}/{dirname}/data/{item['path'].split('/')[-1]}"
            metapath = f"data/{item['path'].split('/')[-1]}"

            try:
                inputs = processor(text=text, return_tensors="pt")
                index = randint(0, len(embeddings)-1)
                speaker_embeddings = torch.tensor(embeddings[index]["xvector"]).unsqueeze(0)
                speech = model.generate_speech(inputs["input_ids"].cuda(), speaker_embeddings.cuda(), vocoder=vocoder)

                sf.write(writepath, speech.cpu().numpy(), samplerate=16000)
                metadata.append({ 'transcription': text, 'path': metapath })

            except Exception as e:
                print(e)

        with open(f"{outpath}/{dirname}/metadata.jsonl", 'w', encoding='utf-8') as outfile:
            for item in metadata:
                outfile.write(f"{json.dumps(item)}\n")

if __name__ == '__main__':
    main()
