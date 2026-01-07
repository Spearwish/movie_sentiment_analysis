import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import IMDBDataset, imdb_collate_fn
from minbpe.regex import RegexTokenizer
from model import TransformerArchitecture


def evaluate(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    TP, FN, FP, TN = 0, 0, 0, 0

    with torch.no_grad():
        for xb, yb in tqdm(data_loader, desc="Evaluating"):
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item()

            # calculate predictions
            preds = (torch.sigmoid(logits) > 0.5).float()

            # confusion matrix - https://en.wikipedia.org/wiki/Confusion_matrix
            # positive prediction
            TP += ((preds == 1.0) & (yb == 1.0)).sum().item()  # positive ground-truth
            FP += ((preds == 1.0) & (yb == 0.0)).sum().item()  # negative ground-truth

            # negative prediction
            TN += ((preds == 0.0) & (yb == 0.0)).sum().item()  # negative ground-truth
            FN += ((preds == 0.0) & (yb == 1.0)).sum().item()  # positive ground-truth

    # accuracy
    acc = ((TP + TN) / (TP + TN + FP + FN)) if (TP + TN + FP + FN) != 0 else 0
    # precisions
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) != 0 else 0
    # recalls
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    # F1-scores
    F1_pos = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) != 0 else 0
    F1_neg = 2 * (NPV * TNR) / (NPV + TNR) if (NPV + TNR) != 0 else 0
    F1_macro = (F1_pos + F1_neg) / 2

    metrics = {
        "loss": total_loss / len(data_loader),
        "accuracy": acc,
        "precision": {"pos": PPV, "neg": NPV},
        "recall": {"pos": TPR, "neg": TNR},
        "f1": {"pos": F1_pos, "neg": F1_neg, "macro": F1_macro},
    }

    return metrics


def get_loss_fn():
    return nn.BCEWithLogitsLoss()


def main():
    # set the seed for reproducibility
    torch.manual_seed(42)
    print(config.DEVICE)

    # initialize tokenizer
    tokenizer = RegexTokenizer()
    tokenizer.load(f"assets/tok{config.VOCAB_SIZE}_{config.TEXT_NORMALIZATION_MODE}.model")

    # load train dataset
    full_dataset = IMDBDataset("datasets/train/", tokenizer)

    # split the dataset into 9:1 ratio -> train:val
    train_size = int(len(full_dataset) * 0.9)  # train_size = config.BATCH_SIZE + skip evaluate() -> overfit test
    val_size = int(len(full_dataset) - train_size)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=imdb_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=imdb_collate_fn)

    # initialize model, optimizer, loss function and logger
    model = TransformerArchitecture(
        config.EMBED_DIM, config.VOCAB_SIZE, config.SEQ_LEN,
        config.N_LAYER, config.N_HEADS, config.FF_DIM, config.DROPOUT
    ).to(config.DEVICE)

    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    loss_fn = get_loss_fn()

    writer = SummaryWriter(log_dir=f"runs/{config.run_name}")

    # initialize validation loss threshold for saving model weights
    best_val_loss = float('inf')

    for epoch in range(config.EPOCHS):
        model.train()  # at the beginning of each epoch ensure that model is in train mode
        epoch_loss = 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}"):
            xb = xb.to(config.DEVICE)
            yb = yb.to(config.DEVICE)

            # zero gradients
            optimizer.zero_grad(set_to_none=True)

            # make forward pass
            logits = model(xb)

            # calculate loss
            loss = loss_fn(logits, yb)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # cumulate the loss
            epoch_loss += loss.item()

        # compute average training loss for the epoch
        train_loss = epoch_loss / len(train_loader)

        # evaluate the model on the validation set
        val_metrics = evaluate(model, val_loader, loss_fn)

        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_metrics["loss"]:.4f} | "
              f"Validation Accuracy: {val_metrics['accuracy'] * 100:.2f}% | Macro F1: {val_metrics['f1']['macro']:.4f}")

        # log training and validation metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

        for group in ["precision", "recall", "f1"]:
            for key, val in val_metrics[group].items():
                writer.add_scalar(f"{group.capitalize()}/val_{key}", val, epoch)

        # save model weights if the validation loss improves
        if val_metrics["loss"] < best_val_loss:
            print("Saving the best model, so far..")
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"./weights/{config.run_name}.pt")

        time.sleep(1)  # an ugly fix of tqdm + print collision -> ghost-bars and distorted console log.

    # flush and close the TensorBoard writer after training to ensure all logs are saved
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
