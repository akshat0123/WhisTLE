from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
# from torch.optim import AdamW
import torch

from bitsandbytes.optim import Adam8bit as AdamW

from tqdm import tqdm

from src import AudioDataset, TextDataset, TextOnlyAdapter, Trainer, Whisper


AUDIO_PATH = '/home/Data/asr/24Feb2023/metadata-training.jsonl'
TEXT_PATH = '/home/Data/asr/24Feb2023/metadata-training.jsonl'
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
    parser.add_argument('--audio_multiplier', type=int, default=1)
    parser.add_argument('--audio_path', type=str, default=AUDIO_PATH)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_acc', type=int, default=1)
    parser.add_argument('--checkpoint_steps', type=int, default=None)
    parser.add_argument('--in_path_toa', type=str, default=None)
    parser.add_argument('--in_path_whisper', type=str, default=None)
    parser.add_argument('--lr_audio', type=float, default=1e-6)
    parser.add_argument('--lr_text', type=float, default=1e-6)
    parser.add_argument('--name', type=str, default='whisper_ft')
    parser.add_argument('--out_path', type=str, default='./data/whisper_ft.pt')
    parser.add_argument('--summary', action='store_true', default=False)
    parser.add_argument('--text_only', action='store_true', default=False)
    parser.add_argument('--text_path', type=str, default=TEXT_PATH)
    parser.add_argument('--total_steps', type=int, default=10000)
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model_type]
    audio_multiplier = args.audio_multiplier
    audio_path = args.audio_path
    batch_size = args.batch_size
    grad_acc = args.grad_acc
    checkpoint_steps = args.checkpoint_steps
    in_path_toa = args.in_path_toa
    in_path_whisper = args.in_path_whisper
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

    # Load whisper model
    whisper = Whisper(model_name).cuda()
    if in_path_whisper is not None:
        whisper.load_state_dict(torch.load(in_path_whisper))

    whisper.train()

    if text_only:
        # Load text-only adapter
        toa = TextOnlyAdapter(
            vocab_size=len(whisper.tokenizer),
            n_dim=whisper.encoder.embed_dim,
            padding_idx=whisper.tokenizer.pad_token_id, 
            in_channels=448, 
            out_channels=1500,
            kernel_size=5
        ).cuda()

        if in_path_toa is not None:
            toa.load_state_dict(torch.load(in_path_toa))

        toa.freeze()
        toa.eval()

    # Load audio data 
    audio_data = AudioDataset(audio_path)
    audio_loader = audio_data.create_loader(whisper.prepare_batch, batch_size=batch_size)
    audio_iterator = iter(audio_loader)

    # Load audio + text transcription trainer
    audio_params = whisper.standard_parameters()
    audio_obj = CrossEntropyLoss()
    audio_opt = AdamW(audio_params, lr_audio)
    audio_sch = OneCycleLR(audio_opt, max_lr=lr_audio, total_steps=total_steps, div_factor=1e2)
    audio_trainer = Trainer(audio_params, audio_opt, audio_obj, audio_sch, grad_acc)

    if text_only:

        # Load text-only data + trainer
        text_data = TextDataset(text_path)
        text_loader = text_data.create_loader(whisper.prepare_batch, batch_size=batch_size)
        text_iterator = iter(text_loader)

        # Load text-only transcription trainer
        text_params = whisper.text_only_parameters()
        text_obj = CrossEntropyLoss()
        text_opt = AdamW(text_params, lr_text)
        text_steps = total_steps // audio_multiplier
        text_sch = OneCycleLR(text_opt, max_lr=lr_text, total_steps=text_steps, div_factor=1e2)
        text_trainer = Trainer(text_params, text_opt, text_obj, text_sch, grad_acc)

    if summary:
        writer = SummaryWriter(f'runs/{name}')

    progress = tqdm(total=total_steps, ncols=160)
    progress_update = False
    audio_current_step = 0
    text_current_step = 0
    update_count = 0
    while audio_current_step < (total_steps * grad_acc):

        for i in range(audio_multiplier):

            # In-domain Audio + text transcription loss
            audio_batch, audio_iterator = get_batch(audio_loader, audio_iterator)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                audio_output = whisper(audio_batch)

            audio_preds = audio_output.logits[:, :, :-1]
            audio_labels = audio_batch.masked_input_ids[:, 1:]
            audio_output = audio_trainer.step(audio_preds, audio_labels, audio_current_step)

            if (audio_current_step + 1) % grad_acc == 0:
                progress_update = True
                progress_postfix = { 'Audio': f"{audio_output['loss']:.2e}" }
                if summary:
                    writer.add_scalar('Audio Loss', audio_output['loss'], audio_current_step+1)

                update_count += 1

            audio_current_step += 1

        if text_only:

            # Out-of-domain text-only transcription loss
            text_batch, text_iterator = get_batch(text_loader, text_iterator)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                toa_encoding = toa(text_batch).encoding
                text_output = whisper(text_batch, encoding=toa_encoding)

            text_preds = text_output.logits[:, :, :-1]
            text_labels = text_batch.masked_input_ids[:, 1:]
            text_output = text_trainer.step(text_preds, text_labels, text_current_step)

            if (text_current_step + 1) % grad_acc == 0:
                progress_postfix['Text'] = f"{text_output['loss']:.2e}"
                if summary:
                    writer.add_scalar('Text Loss', text_output['loss'], text_current_step+1)

            text_current_step += 1

        if progress_update:
            progress.set_postfix(progress_postfix)
            progress.update(update_count)
            progress_update = False
            update_count = 0

    progress.close()

    torch.save(whisper.state_dict(), out_path)


if __name__ == '__main__':
    main()
