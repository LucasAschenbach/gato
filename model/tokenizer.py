"""
Module Description:
-------------------
This module defines the Tokenizer class and its subclasses for different modalities.

Classes:
    - Tokenizer: Base class for all tokenizers.
    - DiscreteTokenizer: Tokenizer for discrete values.
    - ContinuousTokenizer: Tokenizer for continuous values.
    - TextTokenizer: Tokenizer for text data.
    - ImageTokenizer: Tokenizer for image data.
"""

import torch
from transformers import PreTrainedTokenizerFast

class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, values):
        raise NotImplementedError("This method should be overridden by subclasses")

    def detokenize(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses")


class DiscreteTokenizer(Tokenizer):
    def __init__(self, action_path, action_size, offset=0):
        super().__init__()
        discrete_values = self._load_actions(action_path)
        self.value_to_token = {value: i + offset for i, value in enumerate(discrete_values)}
        if len(self.value_to_token) > action_size:
            raise ValueError(f"Too many discrete values. Maximum allowed is {action_size}.")
        self.token_to_value = {i: value for value, i in self.value_to_token.items()}

    def _load_actions(self, action_path):
        with open(action_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def tokenize(self, values):
        try:
            return [self.value_to_token[value] for value in values]
        except KeyError as err:
            raise ValueError(f"Value {err.args[0]} not found in lookup table.")

    def detokenize(self, tokens):
        try:
            return [self.token_to_value[token] for token in tokens]
        except KeyError as err:
            raise ValueError(f"Token {err.args[0]} not found in lookup table.")


class ContinuousTokenizer(Tokenizer):
    def __init__(self, continuous_values_size, mu, m, offset=0):
        super().__init__()
        self.continuous_values_size = continuous_values_size
        self.mu = mu
        self.m = m
        self.offset = offset

    def _mu_law_encode(self, x, mu, m):
        return x.sign() * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu * m)

    def _mu_law_decode(self, x, mu, m):
        sign = torch.sign(x)
        y = x * torch.log1p(mu * m)
        return sign * torch.expm1(torch.abs(y)) / mu

    def tokenize(self, values):
        encoded_values = self._mu_law_encode(values, self.mu, self.m)
        clipped_values = torch.clamp(encoded_values, -1, 1)
        bins = torch.linspace(-1, 1, self.continuous_values_size + 1)
        indices = torch.bucketize(clipped_values, bins) - 1
        return indices + self.offset

    def detokenize(self, tokens):
        indices = tokens - self.offset
        bins = torch.linspace(-1, 1, self.continuous_values_size + 1)
        ecoded_values = (bins[indices] + bins[indices + 1]) / 2
        return self._mu_law_decode(ecoded_values, self.mu, self.m)


class TextTokenizer:
    def __init__(self, vocabulary_path, vocabulary_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.vocabulary = self._load_vocabulary(vocabulary_path)
        if len(self.vocabulary) > self.vocabulary_size:
            raise ValueError(f"Vocabulary size exceeds the limit of {self.vocabulary_size}.")
        self.token_to_value = {i: token for i, token in enumerate(self.vocabulary)}
        self.value_to_token = {token: i for i, token in enumerate(self.vocabulary)}
        self.hf_tokenizer = PreTrainedTokenizerFast(
            vocab=self.value_to_token,
            unk_token="[UNK]"
        )

    def _load_vocabulary(self, vocabulary_path):
        with open(vocabulary_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def tokenize(self, text):
        if isinstance(text, str):
            tokens = self.hf_tokenizer.encode(text, return_tensors="pt")[0]
        else: # Handle batch of texts
            encoded = self.hf_tokenizer(text, padding=True, return_tensors="pt")
            tokens = encoded.input_ids
        return tokens

    def detokenize(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens[0], list):
            decoded = [self.hf_tokenizer.decode(token) for token in tokens]
        else:
            decoded = self.hf_tokenizer.decode(tokens)
        return decoded


class ImageTokenizer:
    def __init__(self, img_patch_size):
        super().__init__()
        self.img_patch_size = img_patch_size

    def tokenize(self, images):
        B, C, H, W = images.shape
        patch_size = self.img_patch_size
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"Image dimensions must be divisible by the patch size {patch_size}")
        unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        patches = unfold(images)  # shape: (B, C * patch_size * patch_size, L)
        L = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(B, L, C, patch_size, patch_size)
        return patches

    def detokenize(self, tokens):
        B, L, C, patch_size, _ = tokens.shape
        patches = tokens.reshape(B, L, C * patch_size * patch_size).transpose(1, 2)
        H = W = int((L * patch_size) ** 0.5)
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        images = fold(patches)
        return images.reshape(B, C, H, W)
