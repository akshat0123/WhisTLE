
from dataclasses import dataclass
from typing import List, Optional
import logging

from torch.nn import Module
from torch import Tensor
import torch

from nemo.collections.asr.models import EncDecMultiTaskModel
logging.getLogger('nemo_logger').setLevel(logging.ERROR)


@dataclass
class CanaryBatch:
    audio_signal: Tensor
    audio_signal_length: Tensor
    input_ids: Tensor
    input_ids_length: Tensor
    input_ids_mask: Tensor
    masked_input_ids: Tensor
    transcripts: List[str]

@dataclass
class CanaryOutput:
    encoding: Tensor
    encoding_mask: Tensor
    logits: Optional[Tensor]=None

class Canary(Module):

    def __init__(self):
        super().__init__()
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-180m-flash')
        self.encoder_decoder_proj = self.model.encoder_decoder_proj
        self.preprocessor = self.model.preprocessor
        self.decoder = self.model.transf_decoder
        self.proj_out = self.model.log_softmax
        self.tokenizer = self.model.tokenizer
        self.encoder = self.model.encoder
        self.prompt = self.model.prompt

        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        self.model.change_decoding_strategy(decode_cfg)

    def standard_parameters(self):
        return list(self.encoder.parameters()) + \
               list(self.encoder_decoder_proj.parameters()) + \
               list(self.decoder.parameters()) + \
               list(self.proj_out.parameters()) 

    def text_only_parameters(self):
        return list(self.decoder.parameters()) + \
               list(self.proj_out.parameters()) 

    def encoder_parameters(self):
        return list(self.encoder.parameters()) + \
               list(self.encoder_decoder_proj.parameters())

    def decoder_parameters(self):
        return list(self.decoder.parameters())

    def language_model_parameters(self):
        return list(self.proj_out.parameters())

    def encode(self, x):
        encoded, encoded_len = self.encoder(
            audio_signal=x.audio_signal, length=x.audio_signal_length
        )
        encoded = encoded.transpose(1, 2)
        encoded = self.encoder_decoder_proj(encoded)
        encoded_mask = self.len_to_mask(encoded_len, max_len=encoded.shape[1]).cuda()
        return CanaryOutput(encoded, encoded_mask)

    def decode(self, x, encoded, decoder_mems_list=None, pos=0,
               return_mems_as_list=False, return_mems=False):
        token_ids = x.input_ids[:, -int(pos!=0):]
        token_ids_mask = x.input_ids_mask[:, -int(pos!=0):]
        decoder_hidden_states = self.decoder._embedding(token_ids, start_pos=pos)

        return self.decoder.decoder(
            decoder_states=decoder_hidden_states,
            decoder_mask=token_ids_mask,
            encoder_states=encoded.encoding,
            encoder_mask=encoded.encoding_mask,
            decoder_mems_list=decoder_mems_list,
            return_mems_as_list=return_mems_as_list,
            return_mems=return_mems,
        )

    def forward(self, x, encoded=None):
        if encoded is None:
            encoded = self.encode(x)

        decoded = self.decode(x, encoded)
        encoded.logits = self.proj_out(hidden_states=decoded).transpose(1, 2)
        return encoded

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def pad_sequence(self, seq, max_len, fill_val=0):
        seq_len = torch.tensor([x.shape[-1] for x in seq])
        seq = torch.cat([torch.cat([
            x[:, :max_len],
            torch.full((1, max_len-x.shape[-1]), fill_val)
        ], dim=-1) for x in seq], dim=0)

        return seq, seq_len

    def len_to_mask(self, x, max_len):
        return torch.cat([torch.cat([
            torch.ones((length)),
            torch.zeros((max_len-length))
        ])[None, :] for length in x], dim=0)

    def get_prompt(self):
        turns = [{
            'role': 'user', 
            'slots': { 
                'decodercontext': '',
                'diarize': 'no',
                'emotion': '<|emo:undefined|>',
                'itn': 'no',
                'pnc': 'no',
                'source_lang': 'en', 
                'target_lang': 'en', 
                'timestamp': 'no',
                'task': 'transcribe'
            }
        }]
        return self.prompt.encode_dialog(turns=turns)["context_ids"].tolist()

    def prepare_batch_audio(self, batch, max_seconds, sample_rate):
        max_signal_length = max_seconds * sample_rate 
        input_signal = [torch.tensor(x.audio)[None, :] for x in batch] 
        input_signal, input_signal_length = self.pad_sequence(input_signal, max_signal_length)
        input_signal = input_signal.cuda()
        input_signal_length = input_signal_length.cuda()
        audio_signal, audio_signal_length = self.preprocessor(
           input_signal=input_signal, length=input_signal_length 
        )
        return audio_signal, audio_signal_length

    def prepare_batch_text(self, batch, max_text_length):
        prompt = self.get_prompt()
        prompt_len = len(prompt)
        input_ids = [
            torch.tensor(
                prompt.copy() + \
                self.tokenizer.text_to_ids(x.transcription, lang_id='en') + \
                [self.tokenizer.eos_id]
            )[None, :] for x in batch
        ]

        input_ids, input_ids_length = self.pad_sequence(
            input_ids, max_text_length, fill_val=self.tokenizer.pad_id
        )

        input_ids = input_ids.cuda()
        input_ids_length = input_ids_length.cuda()
        input_ids_mask = self.len_to_mask(input_ids_length, max_text_length).cuda()
        input_ids_mask[:, :prompt_len-1] = 0
        masked_input_ids = input_ids * input_ids_mask 
        masked_input_ids[masked_input_ids==0] = -100
        masked_input_ids = masked_input_ids.long()

        return input_ids, input_ids_length, input_ids_mask, masked_input_ids

    def prepare_batch(self, batch, max_seconds=15, sample_rate=16000, max_text_length=224):
        audio_signal, audio_signal_length = None, None
        if hasattr(batch[0], 'audio'):
            audio_signal, audio_signal_length = self.prepare_batch_audio(batch, max_seconds, sample_rate)

        input_ids, input_ids_length, input_ids_mask, masked_input_ids = self.prepare_batch_text(batch, max_text_length)
        transcripts = [x.transcription for x in batch]

        return CanaryBatch(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            input_ids=input_ids, input_ids_length=input_ids_length,
            input_ids_mask=input_ids_mask, masked_input_ids=masked_input_ids,
            transcripts=transcripts
        )

    def greedy_search(self, x, encoded):
        prompt_length = x.input_ids.shape[-1]
        batch_size = x.audio_signal.shape[0]
        unfinished = torch.ones((batch_size)).cuda()
        lengths = torch.full((batch_size,), prompt_length).int().cuda()
        end_token = self.tokenizer.eos_id
        decoder_mems_list = None
        pos = 0 

        while torch.sum(unfinished) != 0:
            # Decode and calculate logits
            decoder_mems_list = self.decode(x, encoded, decoder_mems_list, pos, True, True)
            logits = self.proj_out(hidden_states=decoder_mems_list[-1][:, -1:])

            # Get next token, force end token if sequence completed
            next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token * unfinished + end_token * (1 - unfinished)

            # Determine all completed sequences and add to lengths
            unfinished = unfinished.mul((next_token!=end_token).long())
            lengths += (next_token!=end_token).long() 

            # Add new token to running sequences
            x.input_ids = torch.cat([x.input_ids, next_token[:, None]], dim=-1).long()
            x.input_ids_mask = torch.cat([x.input_ids_mask, unfinished[:, None].long()], dim=-1)

            pos += 1
            if pos >= 512: 
                break

        return [x.input_ids[i, prompt_length:lengths[i]] for i in range(batch_size)]

    def transcribe(self, x):
        batch_size = x.audio_signal.shape[0]
        encoded = self.encode(x)
        prompt = torch.tensor(self.get_prompt())
        x.input_ids = torch.stack([prompt for i in range(batch_size)]).cuda()
        x.input_ids_mask = torch.ones(x.input_ids.shape).cuda()

        pred_ids = self.greedy_search(x, encoded)
        return [
            self.tokenizer.ids_to_text(pred_ids[i]) \
            for i in range(len(pred_ids))
        ]


