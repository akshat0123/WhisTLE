from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from typing import Optional

from tqdm import tqdm

from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module, MSELoss
from torch.optim import AdamW
from torch import Tensor
import torch

from src import (
    AudioDataset, Canary, CanaryBatch, RollingCounter, TextOnlyAdapter,
    TextOnlyAdapterBatch, Trainer
)


AUDIO_PATH =  'CommonVoice/metadata-training.jsonl'


def range_test_lr(step, total_steps, min_exp=-12, max_exp=-1):
    return 10**(((abs(max_exp - min_exp) / total_steps) * step) + min_exp)


@dataclass 
class ToaBatch(CanaryBatch, TextOnlyAdapterBatch):
    encoding: Tensor
    encoding_mask: Tensor


class ToaBatchifier:

    def __init__(self, canary):
        self.canary = canary

    def __call__(self, batch):
        batch = self.canary.prepare_batch(batch)
        encoded = self.canary.encode(batch)
        encoding = encoded.encoding
        encoding_mask = encoded.encoding_mask
        encoding_mask = encoding_mask.unsqueeze(-1)

        return ToaBatch(
            audio_signal=batch.audio_signal,
            audio_signal_length=batch.audio_signal_length, encoding=encoding,
            encoding_mask=encoding_mask, input_ids=batch.input_ids,
            input_ids_length=batch.input_ids_length,
            input_ids_mask=batch.input_ids_mask,
            masked_input_ids=batch.masked_input_ids,
            transcripts=batch.transcripts
        )


class ToaLoss(Module):

    def __init__(self, alpha_masked=1.0, alpha_unmasked=1.0, alpha_mask=1.0, alpha_kld=2.0):
        super().__init__()
        self.alpha_unmasked = alpha_unmasked
        self.alpha_masked = alpha_masked
        self.alpha_mask = alpha_mask
        self.alpha_kld = alpha_kld
        self.mse = MSELoss()

    def forward(self, pred, truth, mu, var, mask, pred_mask):
        mse_unmasked = self.mse(pred, truth)
        mse_masked = self.mse(pred*mask, truth*mask)
        mse_mask = self.mse(pred_mask, mask)
        kld = self._kld(mu, var)
        loss = (self.alpha_masked * mse_masked) + \
               (self.alpha_unmasked * mse_unmasked) + \
               (self.alpha_mask * mse_mask) + \
               (self.alpha_kld * kld)
        return loss, mse_masked, mse_unmasked, mse_mask, kld

    def _kld(self, mu, var):
        kl = 0.5 * torch.sum(-1 - var + mu**2 + var.exp(), dim=2)
        return torch.mean(torch.mean(kl, dim=1))


class ToaTrainer(Trainer):

    def __init__(self, params, opt, obj, sch=None):
        super().__init__(params, opt, obj, sch)
        self.metrics['mse_unmasked'] = RollingCounter()
        self.metrics['mse_masked'] = RollingCounter()
        self.metrics['mse_mask'] = RollingCounter()
        self.metrics['kld'] = RollingCounter()

    def calc_loss(self, preds, labels):
        mask = labels.encoding_mask
        labels = labels.encoding
        mu = preds.mu
        var = preds.var
        preds_mask = preds.encoding_mask
        preds = preds.encoding

        loss, mse_masked, mse_unmasked, mse_mask, kld = self.obj(
            preds, labels, mu, var, mask, preds_mask
        )

        self.metrics['mse_unmasked'].add(mse_unmasked.item())
        self.metrics['mse_masked'].add(mse_masked.item())
        self.metrics['mse_mask'].add(mse_mask.item())
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
    parser.add_argument('--audio_path', type=str, default=AUDIO_PATH)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint_steps', type=int, default=None)
    parser.add_argument('--in_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--name', type=str, default='toa')
    parser.add_argument('--out_path', type=str, default='./data/toa.pt')
    parser.add_argument('--summary', action='store_true', default=False)
    parser.add_argument('--total_steps', type=int, default=1000)
    args = parser.parse_args()

    audio_path = args.audio_path
    batch_size = args.batch_size
    checkpoint_steps = args.checkpoint_steps
    in_path = args.in_path
    lr = args.lr
    name = args.name
    out_path = args.out_path
    summary = args.summary
    total_steps = args.total_steps
    checkpoint_steps = checkpoint_steps if checkpoint_steps is not None \
                                        else total_steps + 1

    # Load canary model
    canary = Canary().cuda()
    canary.freeze()
    canary.eval()

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

    if in_path is not None:
        toa.load_state_dict(torch.load(in_path))

    toa.train()

    # Load dataset
    data = AudioDataset(audio_path)
    batchifier = ToaBatchifier(canary)
    loader = data.create_loader(batchifier, batch_size=batch_size)
    iterator = iter(loader)

    batch, iterator = get_batch(loader, iterator)

    # Load text-only encoding trainer
    params = toa.parameters()
    opt = AdamW(params, lr)
    obj = ToaLoss()
    sch = OneCycleLR(
        optimizer=opt, max_lr=lr, total_steps=total_steps, div_factor=1e2,
        final_div_factor=1e2
    )
    trainer = ToaTrainer(params, opt, obj, sch)

    if summary:
        writer = SummaryWriter(f'runs/{name}')

    # Train text-only encoder
    progress = tqdm(total=total_steps, ncols=160)
    current_step = 0
    while current_step < total_steps:

        batch, iterator = get_batch(loader, iterator)
        preds = toa(batch)
        output = trainer.step(preds, batch)

        if (current_step + 1) % checkpoint_steps == 0:
            checkpoint_path = '.'.join(out_path.split('.')[:-1]) + f'_{str(checkpoint_steps)}.pt'
            torch.save(toa.state_dict(), checkpoint_path)

        if summary:
            writer.add_scalar('Loss', output['loss'], current_step+1)
            writer.add_scalar('MSE unmasked', output['mse_unmasked'], current_step+1)
            writer.add_scalar('MSE masked', output['mse_masked'], current_step+1)
            writer.add_scalar('MSE mask', output['mse_mask'], current_step+1)
            writer.add_scalar('KLD', output['kld'], current_step+1)
            writer.add_scalar('LR', output['lr'], current_step+1)

        progress.set_postfix(loss=f"{output['loss']:.2e}",
                             mse_unmasked=f"{output['mse_unmasked']:.2e}",
                             mse_masked=f"{output['mse_masked']:.2e}",
                             mse_mask=f"{output['mse_mask']:.2e}",
                             kld=f"{output['kld']:.2e}",
                             lr=f"{output['lr']:.2e}")
        progress.update(1)

        current_step += 1

    progress.close()

    torch.save(toa.state_dict(), out_path)


if __name__ == '__main__':
    main()

