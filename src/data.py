"""
data.py  —  Penn Treebank loading & tokenization
Uses torchtext's built-in PTB which is small (~5MB) and standard.
Falls back to a tiny synthetic corpus if torchtext PTB is unavailable.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os

# ── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN  = "<pad>"
UNK_TOKEN  = "<unk>"
MASK_TOKEN = "<mask>"
BOS_TOKEN  = "<bos>"
EOS_TOKEN  = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = []

    def build(self, sentences: list[list[str]]) -> None:
        counts = Counter(tok for sent in sentences for tok in sent)
        self.idx2token = SPECIAL_TOKENS[:]
        self.idx2token += [t for t, c in counts.most_common() if c >= self.min_freq]
        self.token2idx = {t: i for i, t in enumerate(self.idx2token)}

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.idx2token[i] for i in ids]

    @property
    def size(self) -> int:
        return len(self.idx2token)

    @property
    def pad_id(self)  -> int: return self.token2idx[PAD_TOKEN]
    @property
    def mask_id(self) -> int: return self.token2idx[MASK_TOKEN]
    @property
    def unk_id(self)  -> int: return self.token2idx[UNK_TOKEN]


def _load_ptb_sentences() -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Try torchtext PTB; fall back to a small synthetic corpus."""
    try:
        from torchtext.datasets import PennTreebank
        from torchtext.data.utils import get_tokenizer

        tokenizer = get_tokenizer("basic_english")

        def _collect(split):
            return [tokenizer(line) for line in PennTreebank(split=split)
                    if line.strip()]

        train = _collect("train")
        valid = _collect("valid")
        test  = _collect("test")
        print(f"[data] PTB loaded  —  train {len(train):,}  valid {len(valid):,}  test {len(test):,}")
        return train, valid, test

    except Exception as e:
        print(f"[data] torchtext PTB unavailable ({e}). Using synthetic corpus.")
        return _synthetic_corpus()


def _synthetic_corpus():
    """Tiny synthetic corpus for offline / CI environments."""
    templates = [
        "the cat sat on the mat",
        "a dog ran through the park",
        "she opened the old wooden door",
        "the quick brown fox jumps over the lazy dog",
        "he read the book in the quiet library",
        "they walked along the river bank",
        "the sun rose above the distant hills",
        "a bird sang in the tall oak tree",
        "the child played with a small red ball",
        "rain fell softly on the empty street",
        "the market was busy every saturday morning",
        "an old man fed the pigeons in the square",
        "she wrote a long letter to her friend",
        "the students listened to the professor carefully",
        "he cooked dinner while listening to music",
    ]
    import random
    random.seed(42)
    all_sents = [[w for w in t.split()] for t in templates]
    # expand to ~1000 sentences by shuffling words
    extended = all_sents * 40
    random.shuffle(extended)
    n = len(extended)
    train = extended[:int(0.8 * n)]
    valid = extended[int(0.8 * n):int(0.9 * n)]
    test  = extended[int(0.9 * n):]
    print(f"[data] Synthetic corpus  —  train {len(train)}  valid {len(valid)}  test {len(test)}")
    return train, valid, test


class TextDataset(Dataset):
    def __init__(self, sentences: list[list[str]], vocab: Vocabulary, seq_len: int):
        self.vocab   = vocab
        self.seq_len = seq_len
        self.data    = self._build(sentences)

    def _build(self, sentences):
        tensors = []
        for sent in sentences:
            ids = self.vocab.encode(sent)
            # truncate or skip sentences that are too short
            if len(ids) < 4:
                continue
            ids = ids[:self.seq_len]
            # pad to seq_len
            ids += [self.vocab.pad_id] * (self.seq_len - len(ids))
            tensors.append(torch.tensor(ids, dtype=torch.long))
        return tensors

    def __len__(self):  return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def get_dataloaders(seq_len: int = 32, batch_size: int = 64, min_freq: int = 2):
    train_sents, valid_sents, test_sents = _load_ptb_sentences()

    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(train_sents)
    print(f"[data] Vocabulary size: {vocab.size:,}")

    train_ds = TextDataset(train_sents, vocab, seq_len)
    valid_ds  = TextDataset(valid_sents, vocab, seq_len)
    test_ds   = TextDataset(test_sents,  vocab, seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    valid_dl  = DataLoader(valid_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dl, valid_dl, test_dl, vocab
