from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import Module, MSELoss
from torch.optim import AdamW
from torch import Tensor
import torch

from tqdm import tqdm

from src import (
    AudioDataset, LRRangeTest, RollingCounter, TextOnlyAdapter,
    TextOnlyAdapterBatch, Trainer, Whisper, WhisperBatch
)


AUDIO_PATH = 'metadata-training.jsonl'
MODEL_TYPES = [ 'tiny', 'base', 'small', 'medium', 'large' ]
MODEL_NAMES = {
    'tiny': 'openai/whisper-tiny', 
    'base': 'openai/whisper-base', 
    'small': 'openai/whisper-small', 
    'medium': 'openai/whisper-medium', 
    'large': 'openai/whisper-large'
}


@dataclass
class ToaBatch(TextOnlyAdapterBatch, WhisperBatch):
    encoding_mask: Tensor
    encoding: Tensor


class ToaBatchifier:

    def __init__(self, whisper, sequence_length=1500, reduction=320):
        self.sequence_length = sequence_length
        self.reduction = reduction
        self.whisper = whisper

    def __call__(self, batch):
        batch = self.whisper.prepare_batch(batch)
        batch_size = len(batch.audio)

        encoding = self.whisper.encode(batch).encoding

        encoding_lengths = [ceil(x.shape[-1]/self.reduction) for x in batch.audio]
        encoding_mask = torch.ones((batch_size, self.sequence_length, 1)).cuda()

        for i in range(len(encoding_lengths)):
            encoding_mask[i, encoding_lengths[i]:, 0] = 0

        return ToaBatch(
            attention_mask=batch.attention_mask, audio=batch.audio,
            encoding=encoding, encoding_mask=encoding_mask,
            input_features=batch.input_features, input_ids=batch.input_ids,
            masked_input_ids=batch.masked_input_ids,
            transcripts=batch.transcripts
        )


class ToaLoss(Module):

    def __init__(self, alpha_masked=1.0, alpha_unmasked=1.0, alpha_kld=2.0):
        super().__init__()
        self.alpha_unmasked = alpha_unmasked
        self.alpha_masked = alpha_masked
        self.alpha_kld = alpha_kld
        self.mse = MSELoss()

    def forward(self, pred, truth, mu, var, mask):
        mse_unmasked = self.mse(pred, truth)
        mse_masked = self.mse(pred*mask, truth*mask)
        kld = self._kld(mu, var)
        loss = (self.alpha_masked * mse_masked) + \
               (self.alpha_unmasked * mse_unmasked) + \
               (self.alpha_kld * kld)
        return loss, mse_masked, mse_unmasked, kld

    def _kld(self, mu, var):
        kl = 0.5 * torch.sum(-1 - var + mu**2 + var.exp(), dim=2)
        return torch.mean(torch.mean(kl, dim=1))


class ToaTrainer(Trainer):

    def __init__(self, params, opt, obj, sch=None, grad_acc=1):
        super().__init__(params, opt, obj, sch, grad_acc)
        self.metrics['mse_unmasked'] = RollingCounter()
        self.metrics['mse_masked'] = RollingCounter()
        self.metrics['kld'] = RollingCounter()

    def calc_loss(self, preds, labels):
        mask = labels.encoding_mask
        labels = labels.encoding
        mu = preds.mu
        var = preds.var
        preds = preds.encoding

        loss, mse_masked, mse_unmasked, kld = self.obj(
            preds, labels, mu, var, mask
        )

        self.metrics['mse_unmasked'].add(mse_unmasked.item())
        self.metrics['mse_masked'].add(mse_masked.item())
        self.metrics['loss'].add(loss.item())
        self.metrics['kld'].add(kld.item())

        return loss


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
    parser.add_argument('--name', type=str, default='toa')
    parser.add_argument('--out_path', type=str, default='./data/toa.pt')
    parser.add_argument('--summary', action='store_true', default=False)
    parser.add_argument('--total_steps', type=int, default=10000)
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
    whisper.freeze()
    whisper.eval()

    # Load text-only adapter
    toa = TextOnlyAdapter(
        vocab_size=len(whisper.tokenizer),
        n_dim=whisper.encoder.embed_dim,
        padding_idx=whisper.tokenizer.pad_token_id, 
        in_channels=448, 
        out_channels=1500,
        kernel_size=5
    ).cuda()

    if in_path is not None:
        toa.load_state_dict(torch.load(in_path))

    toa.train()

    # Load dataset
    data = AudioDataset(audio_path)
    batchifier = ToaBatchifier(whisper)
    loader = data.create_loader(batchifier, batch_size=batch_size)
    iterator = iter(loader)

    # Load text-only encoding trainer
    params = toa.parameters()
    opt = AdamW(params, lr)
    obj = ToaLoss()
    sch = OneCycleLR(
        optimizer=opt, max_lr=lr, total_steps=total_steps, div_factor=1e2,
        final_div_factor=1e2
    )
    trainer = ToaTrainer(params, opt, obj, sch, grad_acc)

    if summary:
        writer = SummaryWriter(f'runs/{name}')

    # Train text-only encoder
    progress = tqdm(total=total_steps, ncols=160)
    current_step = 0
    while current_step < (total_steps * grad_acc):

        batch, iterator = get_batch(loader, iterator)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = toa(batch)

        output = trainer.step(preds, batch, current_step)

        if (current_step + 1) % checkpoint_steps == 0:
            checkpoint_path = '.'.join(out_path.split('.')[:-1]) + f'_{str(checkpoint_steps)}.pt'
            torch.save(toa.state_dict(), checkpoint_path)

        if (current_step + 1) % grad_acc == 0:
            if summary:
                writer.add_scalar('Loss', output['loss'], current_step+1)
                writer.add_scalar('MSE unmasked', output['mse_unmasked'], current_step+1)
                writer.add_scalar('MSE masked', output['mse_masked'], current_step+1)
                writer.add_scalar('KLD', output['kld'], current_step+1)
                writer.add_scalar('LR', output['lr'], current_step+1)

            progress.set_postfix(loss=f"{output['loss']:.4e}",
                                 mse_unmasked=f"{output['mse_unmasked']:.4e}",
                                 mse_masked=f"{output['mse_masked']:.4e}",
                                 kld=f"{output['kld']:.4e}",
                                 lr=f"{output['lr']:.4e}")
            progress.update(1)

        current_step += 1

    progress.close()

    torch.save(toa.state_dict(), out_path)


if __name__ == '__main__':
    main()
