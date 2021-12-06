import sys
import os
import pandas as pd
import pathlib

from typing import Generator
from pathlib import Path
from tqdm import tqdm
from model import ECAPATDNN
from audio_signal import AudioSignal


ecapa = ECAPATDNN()
audio_signal = AudioSignal()
ACCEPTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg")


def file_generator(dir_path: Path) -> Generator[str, None, None]:
    if not Path.is_dir(dir_path):
        raise ValueError("Directory does not exist")

    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for filename in filenames:
            if os.path.splitext(filename)[-1] in ACCEPTED_EXTENSIONS:
                yield os.path.join(dirpath, filename)


def get_band(filepath):
    return str(filepath).split("/")[3]


def get_embedding(filepath):
    # trim sample from 50s to 60s in the song
    signal = audio_signal(file=filepath, transform=True, trim=(50, 60))
    embedding = ecapa(signal.signal)
    return embedding


def get_album(filepath):
    return str(filepath).split("/")[2]


def get_song(filepath):
    return str(filepath).split("/")[-1]


def pipeline(data_path: Path, save_as: Path) -> pd.DataFrame:
    df = pd.DataFrame()

    generator_count = sum(1 for _ in file_generator(data_path))

    files_generator = file_generator(data_path)

    filenames = []
    embeddings = []
    bands = []
    # albums = []
    songs = []
    for filepath in tqdm(files_generator, total=generator_count, position=0, leave=True):
        band = get_band(filepath)
        song = get_song(filepath)
        embedding = get_embedding(filepath)

        filenames.append(filepath)
        embeddings.append(embedding)
        bands.append(band)
        # albums.append(get_album(filepath))
        songs.append(song)

        tqdm.write(f"{band} - {song}")

    df["filename"] = filenames
    df["embedding"] = embeddings
    df["band"] = bands
    # df["album"] = albums
    df["song"] = songs


    if save_as.is_file():
        full_df = pd.read_pickle(save_as)
        df = pd.concat([full_df, df])
        df = df.reset_index(drop=True)
        if full_df.shape[0] > df.shape[0]:
            print(f"Refusing to write to {save_as} due to data loss.")
            sys.exit()
        df.to_pickle(save_as)
        print("Embeddings updated to file: ", save_as)
    else:
        df.to_pickle(save_as)
        print("Embeddings written to file: ", save_as)

    return df


if __name__ == "__main__":
    df_path = Path("full_dataset.pkl")
    genre = "pop"
    data_path = Path("./data/music-dataset/") / genre
    df = pipeline(data_path=data_path, save_as=df_path)
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df["band"].value_counts())
