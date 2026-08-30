from argparse import ArgumentParser
import json

from tqdm import tqdm
import editdistance
import torch

from src import (
    AudioDataset, JsonlDataset, TrigramLM, Whisper, WhisperILME
)


TEXT_PATH = '/home/Data/asr/31Jan2024/metadata-training.jsonl'
TEST_PATH = '/home/Data/asr/31Jan2024/metadata-test.jsonl'
MODEL_PATH = './data/model_1.pt'
MODEL_TYPES = [ 'tiny', 'base', 'small', 'medium', 'large' ]
MODEL_NAMES = {
    'tiny': 'openai/whisper-tiny', 
    'base': 'openai/whisper-base', 
    'small': 'openai/whisper-small', 
    'medium': 'openai/whisper-medium', 
    'large': 'openai/whisper-large'
}


def main():

    parser = ArgumentParser()
    parser.add_argument('--model_type', choices=MODEL_TYPES, type=str, default='base')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--text_path', type=str, default=TEXT_PATH)
    parser.add_argument('--test_path', type=str, default=TEST_PATH)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--beta', type=float, default=0.75)
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model_type]
    model_path = args.model_path
    text_path = args.text_path
    test_path = args.test_path
    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta

    whisper = Whisper(model_name).cuda()
    whisper.load_state_dict(torch.load(model_path))
    whisper.eval()

    data = JsonlDataset(text_path)
    lm = TrigramLM(vocab_size=51865)

    for item in tqdm(data, ncols=120):
        tokens = whisper.tokenizer(item['transcription'])['input_ids']
        lm.add(tokens)

    lm.train()

    model = WhisperILME(whisper, lm, alpha, beta)

    data = AudioDataset(test_path)
    loader = data.create_loader(whisper.prepare_batch, batch_size=batch_size)

    wer_num = 0
    wer_den = 0

    with torch.no_grad():
        for batch in tqdm(loader, ncols=120):
            preds = model(batch)
            truths = batch.transcripts

            for truth, pred in zip(truths, preds):
                truth_tokens = truth.lower().split()
                pred_tokens = pred.lower().split()

                if len(truth_tokens) > 0:
                    wer_num += editdistance.eval(truth_tokens, pred_tokens)
                    wer_den += len(truth_tokens)

    print(f'WER: {wer_num / wer_den:.8f}')


if __name__ == '__main__':
    main()
