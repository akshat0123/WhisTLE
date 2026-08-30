from argparse import ArgumentParser
import string

from tqdm import tqdm
import editdistance
import torch

from src.data import AudioDataset
from src.canary import Canary 


DATA_PATH = 'CommonVoice/metadata-test.jsonl'


def normalize(text):
    return text.strip().lower().translate(str.maketrans('', '', string.punctuation + '¿¡'))


def main():

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size

    model = Canary().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data = AudioDataset(data_path)
    loader = data.create_loader(model.prepare_batch, batch_size=batch_size)

    wer_num = 0
    wer_den = 0

    with torch.no_grad():
        for batch in tqdm(loader, ncols=120):

            preds = model.transcribe(batch)
            truths = batch.transcripts

            for truth, pred in zip(truths, preds):
                truth_tokens = normalize(truth).split()
                pred_tokens = normalize(pred).split()

                if len(truth_tokens) > 0:
                    wer_num += editdistance.eval(truth_tokens, pred_tokens)
                    wer_den += len(truth_tokens)

    print(f'WER: {wer_num / wer_den:.8f}')


if __name__ == '__main__':
    main()

