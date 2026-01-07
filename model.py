import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, dropout):
        super().__init__()

        # multi-head self-attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        # layer norm
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        # feed forward network
        self.linear_stack = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        # self-attention layer
        norm_x = self.layer_norm_1(x)
        attn_output, attn_output_weights = self.multihead_attn(norm_x, norm_x, norm_x, key_padding_mask=padding_mask)

        # residual connection
        x = x + attn_output

        # feed forward layer
        norm_x = self.layer_norm_2(x)
        ffwd_output = self.linear_stack(norm_x)

        # residual connection
        x = x + ffwd_output

        return x


class TransformerArchitecture(nn.Module):
    def __init__(self, embed_dim, vocab_size, seq_len, n_layer, n_heads, ff_dim, dropout):
        super().__init__()
        # embeddings
        self.tok_emb_table = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb_table = nn.Embedding(seq_len + 1, embed_dim)  # adding space for CLS token

        # CLS Token - a learnable vector of size (1, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # transformer backbone
        self.blocks = nn.ModuleList([Block(embed_dim, n_heads, ff_dim, dropout) for _ in range(n_layer)])
        self.layer_norm_f = nn.LayerNorm(embed_dim)

        # binary classification
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, idx):
        B, T = idx.shape

        # token and position embeddings
        tok_emb = self.tok_emb_table(idx)  # (B, T, C)

        # expand the (1,1,C) CLS token parameter to (B, 1, C)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # prepend CLS token to every sequence in the batch
        x = torch.cat((cls_tokens, tok_emb), dim=1)  # (B, T+1, C)

        # create position embeddings for (T+1) tokens
        positions = torch.arange(x.shape[1], device=idx.device)
        pos_emb = self.pos_emb_table(positions)  # (T+1, C)

        x = x + pos_emb  # residual addition

        # create padding mask (True where tokens are equal to token ID 32 -> whitespace character)
        padding_mask = (idx == 32)
        # the CLS token is never padded, (B, 1) mask full of False
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=idx.device)
        # concatenate along sequence dimension
        padding_mask = torch.cat((cls_mask, padding_mask), dim=1)

        # transformer backbone
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        # final normalization
        x = self.layer_norm_f(x)

        # pooling the BERT way, we take the first token - the CLS output
        x = x[:, 0, :]  # (B, C)

        return self.classifier(x)
