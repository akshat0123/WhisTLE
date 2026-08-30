
from transformers import logging
logging.set_verbosity_error()

from argparse import ArgumentParser

from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch

from tqdm import tqdm
import editdistance

from src import AudioDataset, Trainer, WhisperDeepFusion


MODEL_TYPES = [ 'tiny', 'base', 'small', 'medium', 'large' ]
MODEL_NAMES = {
    'tiny': 'openai/whisper-tiny', 
    'base': 'openai/whisper-base', 
    'small': 'openai/whisper-small', 
    'medium': 'openai/whisper-medium', 
    'large': 'openai/whisper-large'
}


def get_batch(loader, iterator):
    try:
        batch = next(iterator)

    except:
        iterator = iter(loader)
        batch = next(iterator)

    return batch, iterator


def main():

    parser = ArgumentParser()
    parser.add_argument('--model_type', choices=MODEL_TYPES, type=str, default='base')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--text_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--total_steps', type=int)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model_type]
    model_path = args.model_path
    train_path = args.train_path
    test_path = args.test_path
    text_path = args.text_path
    batch_size = args.batch_size
    total_steps = args.total_steps
    lr = args.lr

    whisper = WhisperDeepFusion(model_name, text_path, model_path).cuda()
    whisper.train()

    data = AudioDataset(train_path)
    loader = data.create_loader(whisper.prepare_batch, batch_size=batch_size)
    iterator = iter(loader)

    params = list(whisper.parameters())
    opt = AdamW(params, lr)
    obj = CrossEntropyLoss()
    sch = OneCycleLR(opt, max_lr=lr, total_steps=total_steps, div_factor=1e2)
    trainer = Trainer(params, opt, obj, sch)

    progress = tqdm(total=total_steps, ncols=80)
    current_step = 0
    while current_step < total_steps:
        batch, iterator = get_batch(loader, iterator)

        logits = whisper(batch)
        output = trainer.step(
            logits[:, :, :-1], 
            batch.masked_input_ids[:, 1:],
            current_step
        )

        progress.set_postfix(loss=f"{output['loss']:.8f}")
        progress.update(1)

        current_step += 1

    progress.close()

    whisper.eval()
    data = AudioDataset(test_path)
    loader = data.create_loader(whisper.prepare_batch, batch_size=1)

    wer_num = 0
    wer_den = 0

    with torch.no_grad():
        for batch in tqdm(loader, ncols=80):
            preds = whisper.generate(batch)
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
