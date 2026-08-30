from dataclasses import dataclass

from torch.nn import (
    Conv1d, ConvTranspose1d, Embedding, GELU, Module, ModuleList, Sequential, Softplus
)
from torch import randn_like, sqrt, Tensor


class ConvolutionalEncoder(Module):

    def __init__(self, channel_sizes, kernel_size):
        super().__init__()

        self.blocks = ModuleList([
            ConvTranspose1d(channel_sizes[0], channel_sizes[1], kernel_size=1),
        ])

        for i in range(2, len(channel_sizes)-1):
            self.blocks.append(Conv1d( in_channels=channel_sizes[i-1], out_channels=channel_sizes[i], kernel_size=kernel_size, padding='same'))

        self.mu = Conv1d(in_channels=channel_sizes[-2], out_channels=channel_sizes[-1], kernel_size=kernel_size, padding='same')
        self.var = Conv1d(in_channels=channel_sizes[-2], out_channels=channel_sizes[-1], kernel_size=kernel_size, padding='same')

        self.gelu = GELU()
        self.softplus = Softplus()

    def forward(self, x):

        output, residuals = x, []
        for block in self.blocks:
            residual = block(output)
            residuals.append(residual)
            output = self.gelu(residual)

        mu = self.mu(output)
        var = self.softplus(self.var(output))
        z = self.sample(mu, var)
        return mu, var, z, residuals

    def sample(self, mu, var):
        std = sqrt(var + 1e-12)
        z = mu + (std * randn_like(std))
        return z


class ConvolutionalDecoder(Module):

    def __init__(self, channel_sizes, out_channels, kernel_size):
        super().__init__()

        self.blocks = ModuleList()
        for i in range(1, len(channel_sizes)):
            self.blocks.append(ConvTranspose1d(in_channels=channel_sizes[i-1], out_channels=channel_sizes[i], kernel_size=kernel_size, padding=2))

        self.gelu = GELU()

    def forward(self, x, residuals):

        output = x
        for i in range(len(self.blocks)-1):
            output = self.gelu(residuals[i] + self.blocks[i](output))

        return residuals[-1] + self.blocks[-1](output)


class ConvolutionalVAE(Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        encoder_channels = [in_channels] + [out_channels] + [out_channels//(2**i) for i in range(4)]
        decoder_channels = encoder_channels[1:][::-1]
        self.encoder = ConvolutionalEncoder(channel_sizes=encoder_channels, kernel_size=kernel_size)
        self.decoder = ConvolutionalDecoder(channel_sizes=decoder_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        mu, var, z, residuals = self.encoder(x)
        out = self.decoder(z, residuals[::-1])
        return mu, var, out


@dataclass
class TextOnlyAdapterBatch:
    input_ids: Tensor


@dataclass
class TextOnlyAdapterOutput:
    mu: Tensor
    var: Tensor
    encoding: Tensor


class TextOnlyAdapter(Module):

    def __init__(self, vocab_size, n_dim, padding_idx, in_channels,
                 out_channels, kernel_size):
        super().__init__()
        self.embeddings = Embedding(vocab_size, n_dim, padding_idx=padding_idx)
        self.vae = ConvolutionalVAE(in_channels, out_channels, kernel_size)

    def forward(self, batch):
        mu, var, out = self.vae(self.embeddings(batch.input_ids))
        return TextOnlyAdapterOutput(mu, var, out)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
