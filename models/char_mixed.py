#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture for a char-lvl RNN + CNN."""
from models import CharNN
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, LSTM, Bidirectional
from keras.layers import Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class CharMixed(CharNN):
    """Convolutional neural network on char-level data."""

    OPTIMIZER = 'adadelta'

    def model_architecture(self):
        """Setup the model architecture."""
        embedding_size = 128
        conv1_filters = 128
        conv1_kernel_size = 3
        conv2_filters = 256
        conv2_kernel_size = 3
        lstm_units = 256
        hidden_units = 64
        output_units = 6

        inputs    = Input((self.max_len,), name='input')
        embedding = Embedding(
            self.vocab_size, embedding_size, name='embedding')(inputs)

        conv1     = Conv1D(conv1_filters, conv1_kernel_size, strides=1, padding='valid', activation='relu')(embedding)
        conv2     = Conv1D(conv2_filters, conv2_kernel_size, strides=1, padding='valid', activation='relu')(conv1)
        lstm      = Bidirectional(LSTM(lstm_units))(conv2)
        hidden    = Dense(hidden_units, activation='relu')(lstm)
        outputs   = Dense(output_units, activation='sigmoid', name='final')(hidden)
        return Model(inputs=inputs, outputs=outputs)
