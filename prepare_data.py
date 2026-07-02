import os
import requests
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
TINYSTORIES_VAL_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"

def download_file(url, dest):
    if dest.exists():
        print(f"  [skip] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"  Saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

def download_data():
    print("Downloading TinyStories...")
    download_file(TINYSTORIES_URL, DATA_DIR / "TinyStories-train.txt")
    download_file(TINYSTORIES_VAL_URL, DATA_DIR / "TinyStories-valid.txt")

def train_tokenizer(vocab_size=2048):
    tok_path = DATA_DIR / "tokenizer.json"
    if tok_path.exists():
        print(f"  [skip] tokenizer exists at {tok_path}")
        return tok_path

    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    train_file = str(DATA_DIR / "TinyStories-train.txt")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([train_file], trainer)
    tokenizer.save(str(tok_path))
    print(f"  Saved tokenizer to {tok_path} (vocab={tokenizer.get_vocab_size()})")
    return tok_path

def encode_file(tokenizer, txt_path, bin_path, max_tokens=None):
    if bin_path.exists():
        print(f"  [skip] {bin_path.name} ({bin_path.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"  Encoding {txt_path.name}...")
    all_tokens = []
    chunk_size = 5 * 1024 * 1024
    total_tokens = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            enc = tokenizer.encode_batch(chunk.splitlines(keepends=True))
            for e in enc:
                all_tokens.extend(e.ids)
                total_tokens += len(e.ids)
                if max_tokens and total_tokens >= max_tokens:
                    break
            if max_tokens and total_tokens >= max_tokens:
                break
            print(f"    {total_tokens:,} tokens...", end="\r", flush=True)

    tokens_arr = np.array(all_tokens, dtype=np.uint16)
    tokens_arr.tofile(str(bin_path))
    print(f"  Saved {bin_path.name}: {len(tokens_arr):,} tokens ({bin_path.stat().st_size / 1e6:.1f} MB)")

def prepare_data():
    download_data()
    tok_path = train_tokenizer(vocab_size=2048)

    tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"  Tokenizer vocab: {tokenizer.get_vocab_size()}")

    encode_file(tokenizer, DATA_DIR / "TinyStories-train.txt", DATA_DIR / "train.bin")
    encode_file(tokenizer, DATA_DIR / "TinyStories-valid.txt", DATA_DIR / "val.bin")

    print("\nData preparation complete.")
    print(f"  Tokenizer: {tok_path}")
    print(f"  Train:     {DATA_DIR / 'train.bin'}")
    print(f"  Val:       {DATA_DIR / 'val.bin'}")

if __name__ == "__main__":
    prepare_data()