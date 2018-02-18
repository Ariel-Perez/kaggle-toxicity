#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture for a char-lvl RNN."""
import models
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class CharRNN(models.Model):
    """Recurrent neural network on char-level data."""

    OPTIMIZER = 'rmsprop'

    def model_architecture(self):
        """Setup the model architecture."""
        embedding_size = 128
        lstm_units = 128
        hidden_units = 64
        output_units = 6

        inputs    = Input((self.max_len,), name='input')
        embedding = Embedding(
            self.vocab_size, embedding_size, name='embedding')(inputs)
        lstm      = Bidirectional(LSTM(lstm_units))(embedding)
        hidden    = Dense(hidden_units, activation='relu')(lstm)
        outputs   = Dense(output_units, activation='sigmoid', name='final')(hidden)
        return Model(inputs=inputs, outputs=outputs)
