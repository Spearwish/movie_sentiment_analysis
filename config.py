import torch

# hardware for training / inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model architecture hyperparameters
VOCAB_SIZE = 4096  # vocabulary size: 256 base UTF-8 tokens + (VOCAB_SIZE - 256) BPE merges
EMBED_DIM = 32  # dimensionality of the embedding space (C)
FF_DIM = 4 * EMBED_DIM  # hidden layer size of the feed forward network
SEQ_LEN = 768  # maximum sequence length, context window, the model can process
N_HEADS = 8  # number of parallel attention heads
N_LAYER = 6  # number of transformer blocks (depth of the network / backbone)
DROPOUT = 0.1  # probability of zeroing activations to prevent overfitting
TEXT_NORMALIZATION_MODE = "aggressive"  # text preprocessing mode: "standard" vs. "aggressive"
TRUNCATE_END = False # strategy for long sequences: True = keep head, False = keep tail

# training hyperparameters
BATCH_SIZE = 32  # number of training examples processed per iteration
LR = 3e-4  # step size for weight updates used by optimizer
EPOCHS = 25  # total number of complete passes through the training dataset
PADDING_IDX = 32  # token ID used for sequence padding (mapped to whitespace character)

# store a descriptive run name for TensorBoard logs and weight files
run_name = (
    f"vocab{VOCAB_SIZE}_"
    f"emb{EMBED_DIM}_"
    f"ff{FF_DIM}_"
    f"seq{SEQ_LEN}_"
    f"heads{N_HEADS}_"
    f"layers{N_LAYER}_"
    f"drop{DROPOUT}_"
    f"norm{TEXT_NORMALIZATION_MODE}_"
    f"truncEnd{TRUNCATE_END}_"
    f"bs{BATCH_SIZE}_"
    f"lr{LR}"
)
