import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from data import GenericTranslationDataset, BATCH_SIZE


class EncoderDecoderTransformer(nn.Module):

    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            src_key_padding: int,
            tgt_key_padding: int,
            device: torch.device,
            dropout: float = 0.1
    ):
        r"""Transformer with an encoder and decoder for seq2seq models.

        Args:
            d_model: the number of features in the encoder/decoder inputs
            nhead: the number of heads in the multiheadattention models
            num_encoder_layers: number of TransformerEncoderLayers
            num_decoder_layers: number of TransformerDeocderLayers
            src_vocab_size: number of inputs for source embeddings
            tgt_vocab_size: number of inputs for target embeddings
            src_key_padding: the key (value) of padding token ('<pad>') in src
            tgt_key_padding: the key (value) of padding token ('<pad>') in tgt
            dropout: dropout throughout model for training
        """
        super(EncoderDecoderTransformer, self).__init__()
        assert (d_model % nhead == 0), "d_model must be a multiple of nhead"

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout
            ), num_layers=num_encoder_layers
        )

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_positional_encoding = PositionalEncoding(d_model, dropout)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout
            ), num_layers=num_decoder_layers
        )

        self.projection = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self.src_key_padding = src_key_padding
        self.tgt_key_padding = tgt_key_padding
        self.device = device

    def future_token_square_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are
            filled with 1. Unmasked positions are filled with 0. This outputs
            a ByteTensor which, according to PyTorch docs, will mask tokens
            with non-zero values.

            Masking future tokens is only applicable to the decoder.
            https://www.reddit.com/r/MachineLearning/comments/bjgpt2
            /d_confused_about_using_masking_in_transformer/

            torch.triu(..., diagnonal=1) is required to avoid masking the
            current token.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).byte()
        return mask.to(self.device)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        r"""Forward propagate data.

        Shapes:
            src: (S, N, E)
            tgt: (T, N, E)
            tgt_mask: (T, T)
            src_padding_mask: (N, S)
            tgt_padding_mask: (T, S)
        """
        assert (src.shape[1] == tgt.shape[1])
        tgt_seq_len, N = tgt.shape

        src_padding_mask = (src == self.src_key_padding).transpose(0, 1)
        tgt_padding_mask = (tgt == self.tgt_key_padding).transpose(0, 1)
        tgt_future_token_mask = self.future_token_square_mask(tgt_seq_len)

        src_embeds = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embeds = self.src_positional_encoding(src_embeds)
        enc_src = self.encoder(
            src_embeds, src_key_padding_mask=src_padding_mask
        )

        tgt_embeds = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embeds = self.tgt_positional_encoding(tgt_embeds)
        out = self.decoder(
            tgt_embeds, enc_src, tgt_mask=tgt_future_token_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        out = self.projection(out)

        return out


class PositionalEncoding(nn.Module):

    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_local_data() -> Tuple[Tensor, Tensor]:
    # 0 = <sod>
    # 1 = <eod>
    # 2 = <pad>
    # 4 = <unk>
    src_key_padding = 2
    tgt_key_padding = 2

    # "<sod> This is in english right this moment now <eod> <pad>"
    src = torch.tensor([
        [0, 9, 8, 7, 6, 5, 11, 3, 10, 1, src_key_padding],
        [0, 5, 7, 4, 12, 6, 13, 7, 9, 8, 1]
    ]).transpose(0, 1)

    # "<sod> Esto está en inglés ahora mismo <eod> <pad>"
    tgt = torch.tensor([
        [0, 3, 11, 5, 6, 7, 8, 1, tgt_key_padding],
        [0, 9, 8, 7, 3, 1, tgt_key_padding, tgt_key_padding, tgt_key_padding]
    ]).transpose(0, 1)

    return src, tgt


def main():
    # (src, tgt) = get_local_data()
    train_iter = GenericTranslationDataset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 1e-4
    epochs = 100
    batch_size = BATCH_SIZE

    src_key_padding = train_iter.src_pad_idx
    tgt_key_padding = train_iter.tgt_pad_idx
    src_vocab_size = train_iter.src_vocab_len
    tgt_vocab_size = train_iter.tgt_vocab_len

    model = EncoderDecoderTransformer(
        d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        src_key_padding=src_key_padding, tgt_key_padding=tgt_key_padding,
        device=device, dropout=0.1
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_key_padding)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f'[Epoch {epoch} / {epochs}]')

        for (src, tgt) in train_iter:
            # We must right shift the tgt inputs into the decoder, as stated
            # by the
            # original paper Attention is All We Need
            out = model(src, tgt[:-1])

            # Reshape according to use of nn.CrossEntropyLoss
            # input shape: (N, C)
            # tgt shape: (N)
            # collapsing will interweave the training samples but still
            # generate the correct loss
            out = out.reshape(-1, tgt_vocab_size)

            # We don't want to predict the <sod> token, so we left shift the
            # target to ensure it doesn't become a linear mapping
            tgt_loss = tgt[1:].reshape(-1)

            loss = criterion(out, tgt_loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

            optimizer.step()

            if epoch == epochs - 1:
                print(tgt)
                print(tgt.shape)

                softmax = nn.LogSoftmax(dim=1)
                print(softmax(out).argmax(dim=1).reshape(-1, batch_size))
                print(softmax(out).shape)
                print(loss)


if __name__ == '__main__':
    main()
