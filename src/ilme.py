
import math

import numpy as np
import torch

from src.ngram import TrigramLM
from src.whisper import Whisper, WhisperBatch


class WhisperILME:

    def __init__(
            self, 
            whisper: Whisper, 
            language_model: TrigramLM, 
            alpha=0.75, 
            beta=0.75
        ):
        self.whisper = whisper
        self.lm = language_model
        self.alpha = alpha
        self.beta = beta

        self.blank_encoding = self._init_blank_encoding()

    def _init_blank_encoding(self):
        blank_audio = [np.zeros((480000))]
        blank_features = self.whisper.feature_extractor(
            blank_audio, return_tensors='pt', sampling_rate=16000
        ).input_features.cuda()
        blank_encoding = self.whisper.encoder(
            blank_features, output_attentions=True, output_hidden_states=True,
            return_dict=True
        )[0]
        return blank_encoding

    def _get_next_token_lm_logits(self, input_ids):
        lm_outputs = []
        for i in range(input_ids.shape[0]):
            lm_input = input_ids[i][-2:].tolist()
            lm_outputs.append(self.lm(lm_input))

        return torch.cat(lm_outputs)

    def _generate(self, x):

        batch_size = x.input_ids.shape[0]
        unfinished = x.input_ids.new(batch_size).fill_(1)
        end_token = self.whisper.model.config.eos_token_id
        encoding = self.whisper.encode(x).encoding

        while not (x.input_ids[:, -1] == (end_token)).all() and \
              (x.input_ids.shape[1] <= self.whisper.model.config.max_target_positions):

            decoding = self.whisper.decode(x, encoding=encoding)
            next_token_logits = self.whisper.proj_out(decoding)[:, -1, :]

            # Calculate next token logits from internal language model
            blank_encoding = self.blank_encoding.expand(batch_size, -1, -1)
            decoding_ilme = self.whisper.decode(x, encoding=blank_encoding)
            next_token_logits_ilme = self.whisper.proj_out(decoding_ilme)[:, -1, :]
            next_token_logits -= self.alpha * next_token_logits_ilme

            # Calculate next token logits from language model
            lm_output = self._get_next_token_lm_logits(x.input_ids).cuda()
            lm_output = self.beta * (torch.log(lm_output + 1e-12))
            next_token_logits += lm_output[:, 0, :]

            max_scores, next_token = torch.max(next_token_logits, dim=-1)

            # Set end token for all completed sequences
            next_token = next_token * unfinished + end_token * (1 - unfinished)

            # Determine all completed sequences
            unfinished = unfinished.mul((next_token != end_token).long())
            x.input_ids = torch.cat([x.input_ids, next_token[:, None]], dim=-1)

        return x.input_ids 

    def __call__(self, batch):
        batch.input_features = self.whisper.feature_extractor(
            batch.audio, return_tensors='pt', sampling_rate=16000
        ).input_features.cuda()

        input_ids = torch.ones((len(batch.audio), 3), dtype=torch.long,
                               device=self.whisper.model.device)
        input_ids[:, 0] *= (self.whisper.model.config.decoder_start_token_id)
        input_ids[:, 1] *= 50259
        input_ids[:, 2] *= 50359
        batch.input_ids = input_ids
        batch.attention_mask = None

        token_ids = self._generate(batch)
        return self.whisper.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

