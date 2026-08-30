from argparse import ArgumentParser
import string

from tqdm import tqdm
import editdistance
import torch

from src.data import AudioDataset
from src.whisper import Whisper


DATA_PATH = '/home/Data/asr/31Jan2024/metadata-test.jsonl'
MODEL_TYPES = [ 'tiny', 'base', 'small', 'medium', 'large' ]
MODEL_NAMES = {
    'tiny': 'openai/whisper-tiny', 
    'base': 'openai/whisper-base', 
    'small': 'openai/whisper-small', 
    'medium': 'openai/whisper-medium', 
    'large': 'openai/whisper-large'
}


def normalize(text):
    return text.strip().lower().translate(str.maketrans('', '', string.punctuation + '¿¡'))


def main():

    parser = ArgumentParser()
    parser.add_argument('--model_type', choices=MODEL_TYPES, type=str, default='base')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model_type]
    model_path = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size
    limit = args.limit

    model = Whisper(model_name).cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    data = AudioDataset(data_path)
    loader = data.create_loader(model.prepare_batch, batch_size=batch_size)

    wer_num = 0
    wer_den = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, ncols=120):
            preds = model.generate(batch.input_features)
            truths = batch.transcripts

            for truth, pred in zip(truths, preds):
                truth_tokens = normalize(truth).split()
                pred_tokens = normalize(pred).split()

                if len(truth_tokens) > 0:
                    wer_num += editdistance.eval(truth_tokens, pred_tokens)
                    wer_den += len(truth_tokens)

            count += 1
            if limit is not None:
                if count >= limit:
                    break

    print(f'WER: {wer_num / wer_den:.8f}')


if __name__ == '__main__':
    main()
