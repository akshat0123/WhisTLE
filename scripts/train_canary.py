from argparse import ArgumentParser
from functools import partial

from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch

from tqdm import tqdm

from src import AudioDataset, Canary, TextDataset, TextOnlyAdapter, Trainer


AUDIO_PATH = 'CommonVoice/metadata-training.jsonl'
TEXT_PATH = 'CommonVoice/metadata-training.jsonl'


def range_test_lr(step, total_steps, min_exp=-12, max_exp=-1):
    return 10**(((abs(max_exp - min_exp) / total_steps) * step) + min_exp)


def get_batch(loader, iterator):
    try:
        batch = next(iterator)

    except:
        iterator = iter(loader)
        batch = next(iterator)

    return batch, iterator


def main():

    parser = ArgumentParser()
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--audio_multiplier', type=int, default=1)
    parser.add_argument('--audio_path', type=str, default=AUDIO_PATH)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkpoint_steps', type=int, default=None)
    parser.add_argument('--in_path_toa', type=str, default=None)
    parser.add_argument('--in_path_canary', type=str, default=None)
    parser.add_argument('--lr_audio', type=float, default=1e-6)
    parser.add_argument('--lr_text', type=float, default=1e-6)
    parser.add_argument('--name', type=str, default='canary_ft')
    parser.add_argument('--out_path', type=str, default='./data/canary_ft.pt')
    parser.add_argument('--summary', action='store_true', default=False)
    parser.add_argument('--text_only', action='store_true', default=False)
    parser.add_argument('--text_path', type=str, default=TEXT_PATH)
    parser.add_argument('--total_steps', type=int, default=100)
    args = parser.parse_args()

    accumulation = args.accumulation
    audio_multiplier = args.audio_multiplier
    audio_path = args.audio_path
    batch_size = args.batch_size
    checkpoint_steps = args.checkpoint_steps
    in_path_toa = args.in_path_toa
    in_path_canary = args.in_path_canary
    lr_audio = args.lr_audio
    lr_text = args.lr_text
    name = args.name
    out_path = args.out_path
    summary = args.summary
    text_only = args.text_only
    text_path = args.text_path
    total_steps = args.total_steps
    checkpoint_steps = checkpoint_steps if checkpoint_steps is not None \
                                        else total_steps + 1

    # Load canary model
    canary = Canary().cuda()
    if in_path_canary is not None:
        canary.load_state_dict(torch.load(in_path_canary))

    canary.train()

    if text_only:
        # Load text-only adapter
        toa = TextOnlyAdapter(
            vocab_size=canary.tokenizer.vocab_size,
            n_dim=canary.decoder.hidden_size,
            padding_idx=canary.tokenizer.pad_id, 
            in_channels=224,
            out_channels=188,
            kernel_size=5,
            predict_length=True
        ).cuda()

        if in_path_toa is not None:
            toa.load_state_dict(torch.load(in_path_toa))

        toa.freeze()
        toa.eval()

    # Load audio data 
    audio_data = AudioDataset(audio_path)
    audio_loader = audio_data.create_loader(canary.prepare_batch, batch_size=batch_size)
    audio_iterator = iter(audio_loader)

    # Load audio + text transcription trainer
    audio_params = canary.standard_parameters()
    audio_obj = CrossEntropyLoss()
    audio_opt = AdamW(audio_params, lr_audio)
    audio_sch = OneCycleLR(audio_opt, max_lr=lr_audio, total_steps=total_steps, div_factor=1e2)
    audio_trainer = Trainer(audio_params, audio_opt, audio_obj, audio_sch, accumulation)

    if text_only:

        # Load text-only data + trainer
        text_data = TextDataset(text_path)
        text_loader = text_data.create_loader(canary.prepare_batch, batch_size=batch_size)
        text_iterator = iter(text_loader)

        # Load text-only transcription trainer
        text_params = canary.text_only_parameters()
        text_obj = CrossEntropyLoss()
        text_opt = AdamW(text_params, lr_text)
        text_steps = total_steps // audio_multiplier
        text_sch = OneCycleLR(text_opt, max_lr=lr_text, total_steps=text_steps, div_factor=1e2)
        text_trainer = Trainer(text_params, text_opt, text_obj, text_sch, accumulation)

    if summary:
        writer = SummaryWriter(f'runs/{name}')

    progress = tqdm(total=total_steps, ncols=160)
    while audio_trainer.current_step < total_steps:

        for i in range(audio_multiplier):
            # Audio + text transcription loss
            audio_batch, audio_iterator = get_batch(audio_loader, audio_iterator)
            audio_output = canary(audio_batch)
            audio_preds = audio_output.logits[:, :, :-1]
            audio_labels = audio_batch.masked_input_ids[:, 1:]
            audio_output = audio_trainer.step(audio_preds, audio_labels)

            progress_postfix = { 'Audio': f"{audio_output['loss']:.2e}" }
            if summary:
                writer.add_scalar('Audio Loss', audio_output['loss'], audio_trainer.current_step+1)


        if text_only:
            # Text-only transcription loss
            text_batch, text_iterator = get_batch(text_loader, text_iterator)
            toa_encoded = toa(text_batch)
            toa_encoded.encoding_mask = toa_encoded.encoding_mask.squeeze(-1)
            text_output = canary(text_batch, encoded=toa_encoded)
            text_preds = text_output.logits[:, :, :-1]
            text_labels = text_batch.masked_input_ids[:, 1:]
            text_output = text_trainer.step(text_preds, text_labels)

            progress_postfix['Text'] = f"{text_output['loss']:.2e}"
            if summary:
                writer.add_scalar('Text Loss', text_output['loss'], text_trainer.current_step+1)

        inc = audio_trainer.current_step - progress.n
        progress.set_postfix(progress_postfix)
        progress.update(inc)

    progress.close()

    torch.save(canary.state_dict(), out_path)


if __name__ == '__main__':
    main()

