import torch
import torchaudio
import os

from tqdm import tqdm
from typing import Any, Tuple
from torchaudio import transforms as T
from pathlib import Path


class AudioSignal:

    def __init__(self):
        self.signal: torch.Tensor
        self.n_channels: int
        self.dtype: Any
        self.sample_rate: int
        self.filename: Path

    def __call__(self, file: Path, transform: bool, trim: Tuple[int, int]):
        self.filename = file
        self.signal, self.sample_rate = torchaudio.load(file)
        self.dtype = self.signal.dtype

        if transform:
            self.resample()
            self.to_mono()
            self.trim_audio(*trim)

        return self

    def __repr__(self):
        return f"{self.filename} - {self.sample_rate}Hz - {self.dtype}"

    def trim_audio(self, initial: int, final: int):
        self.signal = self.signal[initial:final * self.sample_rate]

    def resample(self, resample_rate: int = 16000):
        resampler = T.Resample(
            self.sample_rate, resample_rate, dtype=self.dtype)
        resampled_waveform = resampler(self.signal)
        self.sample_rate = resample_rate
        self.signal = resampled_waveform

    def to_mono(self):
        self.signal = torch.mean(self.signal, axis=0)


def file_generator(dir_path: Path):
    for file in os.listdir(dir_path):
        if not file.endswith(".wav"):
            continue
        yield os.path.join(dir_path, file)

