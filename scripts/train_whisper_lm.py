from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch

from tqdm import tqdm

from src import AudioDataset, Trainer, Whisper


AUDIO_PATH = '/home/Data/asr/24Feb2023/metadata-training.jsonl'
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
    parser.add_argument('--audio_path', type=str, default=AUDIO_PATH)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_acc', type=int, default=1)
    parser.add_argument('--checkpoint_steps', type=int, default=None)
    parser.add_argument('--in_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', type=str, default='whisper_lm')
    parser.add_argument('--out_path', type=str, default='./data/whisper_lm.pt')
    parser.add_argument('--summary', action='store_true', default=False)
    parser.add_argument('--total_steps', type=int, default=1000)
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model_type]
    audio_path = args.audio_path
    batch_size = args.batch_size
    grad_acc = args.grad_acc
    checkpoint_steps = args.checkpoint_steps
    in_path = args.in_path
    lr = args.lr
    name = args.name
    out_path = args.out_path
    summary = args.summary
    total_steps = args.total_steps
    checkpoint_steps = checkpoint_steps if checkpoint_steps is not None \
                                        else total_steps + 1

    # Load whisper model
    whisper = Whisper(model_name).cuda()

    if in_path is not None:
        whisper.load_state_dict(torch.load(in_path))

    whisper.train()
    whisper.freeze()
    whisper.unfreeze_language_model()

    # Load dataset
    data = AudioDataset(audio_path)
    loader = data.create_loader(whisper.prepare_batch, batch_size=batch_size)
    iterator = iter(loader)

    # Load trainer
    params = whisper.language_model_parameters()
    trainer = Trainer(params, AdamW(params, lr), CrossEntropyLoss())

    if summary:
        writer = SummaryWriter(f'runs/{name}')

    # Train whisper language modeling layer
    progress = tqdm(total=total_steps, ncols=160)
    current_step = 0
    while current_step < (total_steps * grad_acc):

        batch, iterator = get_batch(loader, iterator)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = whisper(batch)

        output = trainer.step(
            preds.logits[:, :, :-1], 
            batch.masked_input_ids[:, 1:],
            current_step
        )

        if (current_step + 1) % checkpoint_steps == 0:
            checkpoint_path = '.'.join(out_path.split('.')[:-1]) + f'_{str(checkpoint_steps)}.pt'
            torch.save(whisper.state_dict(), checkpoint_path)

        if (current_step + 1) % grad_acc == 0:
            if summary:
                writer.add_scalar('LM Head Loss', output['loss'], current_step+1)

            progress.set_postfix(loss=f"{output['loss']:.8f}")
            progress.update(1)

        current_step += 1

    progress.close()

    torch.save(whisper.state_dict(), out_path)


if __name__ == '__main__':
    main()
