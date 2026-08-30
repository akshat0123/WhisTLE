from dataclasses import dataclass
from typing import List
import json, random

from torch.utils.data import DataLoader, Dataset
import librosa, torch
import numpy as np


class JsonlDataset(Dataset):

    def __init__(self, path, limit=None, shuffle=False):
        self.metadata = [json.loads(x) for x in open(path, 'r').readlines()]
        self.root = '/'.join(path.split('/')[:-1])

        if limit:
            self.metadata = self.metadata[:limit]

        if shuffle:
            random.shuffle(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return self.metadata[idx]

    def __iter__(self):
        return iter(self.metadata)

    def create_loader(self, collate_fn, batch_size=1, shuffle=False):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate_fn
        )


@dataclass
class Audio:
    audio: np.ndarray
    transcription: str


class AudioDataset(JsonlDataset):

    def __getitem__(self, idx):
        path = f"{self.root}/{self.metadata[idx]['path']}"
        waveform, sample_rate = librosa.load(path, sr=16000)
        transcription = self.metadata[idx]['transcription']
        return Audio(audio=waveform, transcription=transcription)


@dataclass
class Text:
    transcription: str


class TextDataset(JsonlDataset):

    def __getitem__(self, idx):
        transcription = self.metadata[idx]['transcription']
        return Text(transcription=transcription)

