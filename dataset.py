import glob
import os
import unicodedata

import regex as re
import torch
from torch.utils.data import Dataset

import config

# dictionary mapping used for text preprocessing
replace_dict = {
    "‘": "'", "’": "'", "`": "'", "´": "'",
    '“': '"', '”': '"',
    "–": "-", "—": "-",
    "…": "...",
    "₤": "£",
    "⁄": "/",
    "\xad": "",  # soft hyphen -> delete
}

# regex pattern to detect unwanted characters in text
NOISE_PATTERN = re.compile(r'[\x00-\x1F\x7F-\x9F\uF000-\uF0FF]')


def normalize_text(text, mode="standard"):
    # the required normalization minimum
    # NFC - merge characters where possible 'A' + '´' becomes 'Á', decomposed characters influence tokenizer.
    text = unicodedata.normalize('NFC', text)

    # map duplicates: "‘", "’", "`", "´" -> "'"
    for src, tgt in replace_dict.items():
        text = text.replace(src, tgt)

    # remove noise: '\x80', '\x84' -> ''
    text = NOISE_PATTERN.sub('', text)

    # remove redundant whitespace:
    text = re.sub(r'\s+', ' ', text).strip()

    if mode == "aggressive":
        # convert the text to lower case
        text = text.lower()

        # remove accent
        text = unicodedata.normalize('NFD', text)
        text = "".join([c for c in text if not unicodedata.category(c) == 'Mn'])

    return text


def normalize_sequence_length(x):
    if len(x) >= config.SEQ_LEN:
        if config.TRUNCATE_END:
            # truncate the end, keeping first SEQ_LEN tokens
            return x[:config.SEQ_LEN]
        else:
            # truncate the beginning, keeping last SEQ_LEN tokens
            return x[-config.SEQ_LEN:]

    else:
        # post-pad (add spaces to the end)
        pad_len = config.SEQ_LEN - len(x)

        padding = torch.full((pad_len,), config.PADDING_IDX, dtype=torch.long)
        return torch.cat([x, padding])


def imdb_collate_fn(batch):
    # unpack the batch into inputs and targets
    inputs, targets = zip(*batch)

    # normalize sequence length for each input
    processed_inputs = [normalize_sequence_length(x) for x in inputs]

    # stack inputs into (B, T) and labels into (B, 1)
    return torch.stack(processed_inputs), torch.stack(targets)


class IMDBDataset(Dataset):
    def __init__(self, folder_path, tokenizer):
        self.tokenizer = tokenizer
        self.file_paths = []
        self.labels = []

        # collect file paths and labels from 'pos' and 'neg' subfolders
        for label_str, label_int in [('pos', 1.0), ('neg', 0.0)]:
            path = os.path.join(folder_path, label_str, "*.txt")
            files = glob.glob(path)
            self.file_paths.extend(files)
            self.labels.extend([label_int] * len(files))

    def __len__(self):
        # total number of reviews in the dataset
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # load the review text - small CPU overhead - could be moved to init along with normalization and tokenization
        with open(path, 'r', encoding='utf-8') as f:
            review = f.read()

        # normalize text and encode into tokens
        tokens = self.tokenizer.encode(normalize_text(review, mode=config.TEXT_NORMALIZATION_MODE))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor([label], dtype=torch.float)
