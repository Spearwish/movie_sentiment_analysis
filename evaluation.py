"""
This file tests the model architecture and configuration against a test dataset.
Run this script only once, after completing training and hyperparameter tuning, to obtain the final test set score.
"""
import torch
from torch.utils.data import DataLoader

import config
from dataset import IMDBDataset, imdb_collate_fn
from minbpe.regex import RegexTokenizer
from model import TransformerArchitecture
from train import evaluate, get_loss_fn

run_final_test = False

if run_final_test:
    # initialize tokenizer and test dataset
    tokenizer = RegexTokenizer()
    tokenizer.load(f"assets/tok{config.VOCAB_SIZE}_{config.TEXT_NORMALIZATION_MODE}.model")

    # load test data and create DataLoader
    test_dataset = IMDBDataset("datasets/test/", tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=imdb_collate_fn)

    # initialize model and load model weights
    model = TransformerArchitecture(
        config.EMBED_DIM, config.VOCAB_SIZE, config.SEQ_LEN,
        config.N_LAYER, config.N_HEADS, config.FF_DIM, config.DROPOUT
    ).to(config.DEVICE)

    model.load_state_dict(torch.load(f"./weights/{config.run_name}.pt", weights_only=True))
    model.eval()

    # run the evaluation on the test data
    test_metrics = evaluate(model, test_loader, get_loss_fn())
    print(test_metrics)
