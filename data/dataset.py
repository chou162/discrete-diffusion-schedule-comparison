"""
Loads Penn Treebank sentences from disk and builds a vocabulary
capped at the 1000 most frequent tokens. Falls back to a small
synthetic corpus if PTB files aren't found.
"""

import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
import torch


# ── Special tokens ─────────────────────────────────────────────────────────────
PAD_TOKEN   = "<pad>"
UNK_TOKEN   = "<unk>"
MASK_TOKEN  = "<mask>"
BOS_TOKEN   = "<bos>"
EOS_TOKEN   = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    """Simple word-level vocabulary with encode/decode."""

    def __init__(self, min_freq: int = 2, max_size: int = 1000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2id: dict[str, int] = {}
        self.id2token: list[str] = []

    def build(self, sentences: list[list[str]]) -> None:
        counts = Counter(tok for sent in sentences for tok in sent)
        top_tokens = [tok for tok, _ in counts.most_common(self.max_size)
                      if tok not in SPECIAL_TOKENS]
        self.id2token = SPECIAL_TOKENS + top_tokens
        self.token2id = {tok: i for i, tok in enumerate(self.id2token)}

    def __len__(self) -> int:
        return len(self.id2token)

    # Convenience id properties
    @property
    def pad_id(self)  -> int: return self.token2id[PAD_TOKEN]
    @property
    def unk_id(self)  -> int: return self.token2id[UNK_TOKEN]
    @property
    def mask_id(self) -> int: return self.token2id[MASK_TOKEN]
    @property
    def bos_id(self)  -> int: return self.token2id[BOS_TOKEN]
    @property
    def eos_id(self)  -> int: return self.token2id[EOS_TOKEN]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token[i] if i < len(self.id2token) else UNK_TOKEN for i in ids]


class PTBDataset(Dataset):
    """
    Penn Treebank sentences, padded/truncated to `max_len` tokens.
    Each item is a LongTensor of token ids.
    """

    def __init__(
        self,
        sentences: list[list[str]],
        vocab: Vocabulary,
        max_len: int = 32,
    ):
        self.vocab   = vocab
        self.max_len = max_len
        self.data    = self._encode(sentences)

    def _encode(self, sentences: list[list[str]]) -> list[torch.Tensor]:
        encoded = []
        for sent in sentences:
            ids = self.vocab.encode(sent[:self.max_len])
            # Pad or truncate to max_len
            if len(ids) < self.max_len:
                ids = ids + [self.vocab.pad_id] * (self.max_len - len(ids))
            encoded.append(torch.tensor(ids, dtype=torch.long))
        return encoded

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on whitespace."""
    return text.lower().split()


def load_ptb(
    max_len: int = 32,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, Vocabulary]:
    """
    Downloads Penn Treebank via HuggingFace datasets, builds vocab, returns
    (train_loader, val_loader, vocab).

    Falls back to a small synthetic corpus if the dataset is unavailable
    (useful for quick local testing without internet access).
    """
    sentences = _load_sentences()

    vocab = Vocabulary(min_freq=2, max_size=1000)
    vocab.build([_tokenize(s) for s in sentences])

    tokenized = [_tokenize(s) for s in sentences]
    dataset   = PTBDataset(tokenized, vocab, max_len=max_len)

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[data] vocab size: {len(vocab)}")
    print(f"[data] train: {len(train_ds)} | val: {len(val_ds)} | seq_len: {max_len}")

    return train_loader, val_loader, vocab


def _load_sentences() -> list[str]:
    """Load PTB from local files downloaded from GitHub."""
    import os
    paths = ["data/ptb/train.txt", "data/ptb/valid.txt"]
    sents = []
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                sents += [l.strip() for l in f if len(l.strip().split()) >= 5]
    if sents:
        print(f"[data] loaded PTB from disk: {len(sents)} sentences")
        return sents
    print("[data] PTB not found on disk, using synthetic corpus")
    return _synthetic_corpus()


def _synthetic_corpus() -> list[str]:
    """500 templated sentences as a fallback corpus for offline testing."""
    templates = [
        "the {a} {v} the {b} in the {p}",
        "a {a} {n} {v} with the {b}",
        "the {n} said the {a} {b} was {p}",
        "investors {v} shares after the {a} report",
        "the company {v} its {a} {n} last year",
        "analysts said the {a} market {v} significantly",
        "the {a} government {v} new economic policies",
        "trading in {a} stocks {v} on the exchange",
        "the {a} bank {v} interest rates this month",
        "officials {v} the {a} proposal in committee",
    ]
    adjectives = ["large", "small", "new", "old", "major", "local", "national",
                  "federal", "annual", "quarterly", "recent", "current"]
    nouns      = ["bank", "company", "market", "stock", "fund", "report", "plan",
                  "group", "firm", "unit", "board", "rate", "price", "year"]
    verbs      = ["said", "reported", "increased", "declined", "announced",
                  "approved", "released", "issued", "sold", "bought"]
    places     = ["market", "exchange", "sector", "region", "quarter"]

    import random
    rng = random.Random(0)
    corpus = []
    for _ in range(500):
        t = rng.choice(templates)
        corpus.append(t.format(
            a=rng.choice(adjectives),
            b=rng.choice(nouns),
            n=rng.choice(nouns),
            v=rng.choice(verbs),
            p=rng.choice(places),
        ))
    return corpus
