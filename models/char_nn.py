#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture for a char-lvl RNN."""
import models
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from metrics import f_score


class CharNN(models.Model):
    """Neural network on char-level data."""

    INTERNAL_VARIABLES = ["vocab_size", "max_len", "tokenizer"]

    def __init__(self, vocab_size, max_len):
        """Initialize."""
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.model = self.model_architecture()
        self.model.compile(
            optimizer=self.OPTIMIZER,
            loss='binary_crossentropy',
            metrics=[f_score]
        )

        self.tokenizer = Tokenizer(
            num_words=vocab_size,
            filters='\t\n',
            lower=True,
            char_level=True
        )

    def model_architecture(self):
        """Setup the model architecture."""
        raise NotImplementedError()

    def fit_preprocessor(self, data):
        """Prepare the tokenizer for use."""
        if self.tokenizer.document_count == 0:
            for i in range(len(data)):
                x, y = data[i]
                self.tokenizer.fit_on_texts(x.comment_text)

    def preprocess(self, inputs):
        """Preprocess using the tokenizer."""
        tokenized_seqs = self.tokenizer.texts_to_sequences(inputs.comment_text)
        padded_seqs = pad_sequences(
            tokenized_seqs, maxlen=self.max_len,
            padding='pre', truncating='post',
            value=0.)

        return padded_seqs
