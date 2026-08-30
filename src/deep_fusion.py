
from dataclasses import asdict, dataclass

from tqdm import tqdm
import torch

from src.data import JsonlDataset
from src.ngram import TrigramLM
from src.whisper import Whisper, WhisperBatch


class FusionModel(torch.nn.Module):

    def __init__(self, d_vocab, d_hidden, d_in):
        super().__init__()
        self.proj_in = torch.nn.Linear(d_hidden, d_in)
        self.proj_out = torch.nn.Linear(d_in, d_vocab)
        self.sigmoid = torch.nn.Sigmoid()
        self.gelu = torch.nn.GELU()

        torch.nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.proj_out.bias, -5.0)

    def forward(self, lm_probs: torch.Tensor, decoding: torch.Tensor):

        # Calculate lm probability weights and reweight whisper logits
        lm_prob_weights = self.sigmoid(self.proj_out(self.gelu(self.proj_in(decoding))))
        reweighted_lm_probs = lm_prob_weights * torch.log(lm_probs + 1e-12)
        return reweighted_lm_probs


@dataclass
class WhisperDeepFusionBatch(WhisperBatch):
    lm_probs: torch.Tensor


class WhisperDeepFusion(torch.nn.Module):

    def __init__(self, model_name, text_path, model_path=None):
        super().__init__()

        self.model_name = model_name
        self.text_path = text_path

        self.whisper = Whisper(model_name)
        self.whisper.eval()

        self.lm = TrigramLM(vocab_size=self.whisper.vocab_size, smooth=True)
        self.fusion_model = FusionModel(
            self.whisper.vocab_size, self.whisper.d_model, 128
        )

        if model_path is not None:
            self.whisper.load_state_dict(torch.load(model_path))

        self._train_lm()

    def _train_lm(self):
        data = JsonlDataset(self.text_path)
        for item in tqdm(data, ncols=80, desc='Training language model'):
            tokens = self.whisper.tokenizer(item['transcription'])['input_ids']
            self.lm.add(tokens)

        self.lm.train()

    def _freeze_whisper(self):
        for param in self.whisper.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.whisper.train(mode=False)
        self._freeze_whisper()
        self.fusion_model.train(mode)

    def parameters(self):
        return self.fusion_model.parameters()

    def _get_next_token_lm_output(self, input_ids):
        lm_outputs = []
        for i in range(input_ids.shape[0]):
            lm_input = input_ids[i][-2:].tolist()
            lm_outputs.append(self.lm(lm_input))

        return torch.cat(lm_outputs).cuda()

    def _get_lm_output(self, input_ids):
        lm_output = []
        for i in range(input_ids.shape[0]):
            lm_inputs = input_ids[i].tolist()
            lm_single_output = []

            for j in range(len(lm_inputs)):
                lm_input = lm_inputs[:j]
                lm_single_output.append(*self.lm(lm_input))

            lm_output.append(torch.cat(lm_single_output))
        lm_output = torch.stack(lm_output).cuda()
        return lm_output

    def prepare_batch(self, batch):
        batch = self.whisper.prepare_batch(batch)
        lm_output = self._get_lm_output(batch.input_ids)
        return WhisperDeepFusionBatch(**asdict(batch), lm_probs=lm_output)

    def generate(self, batch):
        input_features = self.whisper.feature_extractor(
            batch.audio, return_tensors='pt', sampling_rate=16000
        ).input_features.cuda()

        input_ids = torch.ones(
            (len(batch.audio), 3), dtype=torch.long, 
            device=self.whisper.model.device
        )
        input_ids[:, 0] *= (self.whisper.model.config.decoder_start_token_id)
        input_ids[:, 1] *= 50259
        input_ids[:, 2] *= 50359
        batch.input_ids = input_ids
        batch.attention_mask = None

        unfinished = batch.input_ids.new(batch.input_ids.shape[0]).fill_(1)
        end_token = self.whisper.model.config.eos_token_id
        encoding = self.whisper.encode(batch).encoding

        while not (batch.input_ids[:, -1] == (end_token)).all() and \
              (batch.input_ids.shape[1] <= self.whisper.model.config.max_target_positions):

            decoding = self.whisper.decode(batch, encoding=encoding)[:, -1, :]
            next_token_logits = self.whisper.proj_out(decoding)

            # Calculate next token logits from language model
            lm_output = self._get_next_token_lm_output(batch.input_ids)[:, -1, :]

            # Reweight the whisper logits
            reweighted_lm_output = self.fusion_model(lm_output, decoding)
            next_token_logits += reweighted_lm_output

            # Calculate max scoring token
            max_scores, next_token = torch.max(next_token_logits, dim=-1)

            # Set end token for all completed sequences
            next_token = next_token * unfinished + end_token * (1 - unfinished)

            # Determine all completed sequences
            unfinished = unfinished.mul((next_token != end_token).long())
            batch.input_ids = torch.cat([batch.input_ids, next_token[:, None]], dim=-1)

        return self.whisper.tokenizer.batch_decode(batch.input_ids, skip_special_tokens=True)

    def forward(self, batch: WhisperDeepFusionBatch):
        with torch.no_grad():
            encoding = self.whisper.encode(batch).encoding
            decoded = self.whisper.decode(batch, encoding)
            logits = self.whisper.proj_out(decoded)

        lm_probs = self.fusion_model(batch.lm_probs, decoded)
        logits += lm_probs

        return logits.permute(0, 2, 1)
