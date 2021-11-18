import os
import pandas as pd

from typing import Generator
from pathlib import Path
from tqdm import tqdm
from src.model import ECAPATDNN
from src.audio_signal import AudioSignal


ecapa = ECAPATDNN()
audio_signal = AudioSignal()
ACCEPTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg")


def file_generator(dir_path: str) -> Generator[str, None, None]:
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for filename in filenames:
            if os.path.splitext(filename)[-1] in ACCEPTED_EXTENSIONS:
                yield os.path.join(dirpath, filename)


def get_band(filepath):
    return str(filepath).split("/")[1]


def get_embedding(filepath):
    # trim sample from 50s to 60s in the song
    signal = audio_signal(file=filepath, transform=True, trim=(50, 60))
    embedding = ecapa(signal.signal)
    return embedding


def get_album(filepath):
    return str(filepath).split("/")[2]


def get_song(filepath):
    return str(filepath).split("/")[3]


def pipeline(data_path: Path):
    df = pd.DataFrame()

    files_generator = file_generator(data_path)

    filenames = []
    embeddings = []
    bands = []
    albums = []
    songs = []
    for filepath in tqdm(files_generator):
        filenames.append(filepath)
        embeddings.append(get_embedding(filepath))
        bands.append(get_band(filepath))
        albums.append(get_album(filepath))
        songs.append(get_song(filepath))

    df["filename"] = filenames
    df["embedding"] = embeddings
    df["band"] = bands
    df["album"] = albums
    df["song"] = songs

    df.to_pickle("songs_embeddings.pkl")

    return df


if __name__ == "__main__":
    data_path = Path("./data/metallica/")
    df = pipeline(data_path=data_path)
    print(df.head())
