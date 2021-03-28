# PyTorch-Transformer
An example repo that builds a seq2seq machine translation transformer model on a small, locally created, dataset.

## Transformer Architecture

### Encoder Modules

1. Embeddings
2. Positional Encoding
3. TransformerEncoderLayers

### Decoder Modules

1. Embeddings
2. Positional Encoding
3. TransformerDecoderLayers
4. Linear Projection

### Token Masking

There are two general types of masking that can occur in a transformer model for NLP: future token masking and padding masking.

**Future Token Masking**

* `(src/mem/tgt)_mask` in PyTorch modules

Arguably the more important masking of the two types, future token masking ensures the model is unable to look at future tokens in the sequence during training. This masking does not allow the model to attend to unseen tokens at a given timestep.

PyTorch's documentation states that users can use `Float/Bool/ByteTensor` objects to perform this masking. This repo uses a `ByteTensor` to prevent attending to future tokens. Nonzero values are not allowed to be attended to while zero values are allowed. This creates an upper trinagular matrix shifted right by one column, therefore allowing the current token to attend to the current token.

```python
>>> torch.triu(torch.ones(5, 5), diagonal=1).byte()
tensor([[0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)
```

**Padding Masking**

* `(src/mem/tgt)_key_padding_mask` in PyTorch modules

Padding masking is an artifact of batching. Sequences are not guaranteed to be the same length within any given batch, so shorter sequences are forced to be padded. This padding is applied upon the `<pad>` tokens.

In similar nature to the `(src/mem/tgt)_mask` above, this can be of `Float/Bool/ByteTensor`. In `BoolTensor` scenarios, values of `False` are not masked while `True` values are masked.

```python
>>> src_key_padding = 1
>>> src
tensor([[ 4,  4,  4,  4],
        [16, 14, 10, 13],
        [ 5, 23,  8, 12],
        [27,  6, 18,  5],
        [ 7, 29, 17,  7],
        [33,  5, 34, 19],
        [30, 11, 22, 20],
        [28,  8, 25, 35],
        [36, 32, 24, 38],
        [ 2,  6, 37,  2],
        [ 3, 31, 26,  3],
        [ 1, 21,  9,  1],
        [ 1,  2, 15,  1],
        [ 1,  3,  2,  1],
        [ 1,  1,  3,  1]])
>>> src_key_padding_mask = (src == src_key_padding)
tensor([[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [ True, False, False,  True],
        [ True, False, False,  True],
        [ True, False, False,  True],
        [ True,  True, False,  True]])
```

## Training

### Token Shifts

The decoder relies upon the right-shift of its input tokens throughout training, while the output is compared to a left-shifted target. The right-shift, as mentioned in the paper *Attention is All You Need*, prevents the transformer from learning a simple mapping of token to token and removes the `<eod>` token from the longest sequences in a batch. This right-shift is seen in the training loop. The output is then reshaped to appropriate dimensionality for use with `nn.CrossEntropyLoss`

```python
out = model(src, tgt[:-1])
out = out.reshape(-1, tgt_vocab_size)
```

The left-shift occurs, again during training, as the expected output. The left-shift removes the `<sod>` token, the opposite of the right-shift. This reshaping is again seen in the training loop.

```python
tgt_loss = tgt[1:].reshape(-1)
```

```python
>>> tgt.shape
torch.Size([5, 3])
>>> tgt
tensor([[ 5,  5,  5],
        [ 8,  7, 11],
        [ 2, 19, 10],
        [23,  6,  2],
        [ 1,  4,  1]])
>>> 
>>> tgt[:-1].shape
torch.Size([4, 3])
>>> tgt[:-1]
tensor([[ 5,  5,  5],
        [ 8,  7, 11],
        [ 2, 19, 10],
        [23,  6,  2]])
>>>
>>> tgt[1:].shape
torch.Size([4, 3])
>>> tgt[1:]
tensor([[ 8,  7, 11],
        [ 2, 19, 10],
        [23,  6,  2],
        [ 1,  4,  1]])
```
> `batch_size=3`
