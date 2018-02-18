#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains the Data class for specifying a thread-safe Keras Sequence."""
import numpy as np
import copy
from keras.utils import Sequence


class Data(Sequence):
    """This object is used for fitting models."""

    def __init__(self, **kwargs):
        """Initialize the sequence with a Pandas Dataframe."""
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert 'data' in kwargs
        assert 'batch_size' in kwargs
        assert 'features' in kwargs
        assert 'labels' in kwargs

    def __len__(self):
        """The length of this sequence."""
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Get a batch from this data."""
        batch = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = batch[self.features]
        batch_y = batch[self.labels]

        return batch_x, batch_y

    def split(self, fraction):
        """
        Split the data according to the given fraction.

        WARNING: If subclassing, make sure this function is overriden.
        """
        split_index = int(len(self.data) * fraction)
        df1 = self.data.iloc[:split_index, :]
        df2 = self.data.iloc[split_index:, :]

        kwargs1 = copy.deepcopy(self.kwargs)
        kwargs1['data'] = df1

        kwargs2 = copy.deepcopy(self.kwargs)
        kwargs2['data'] = df2

        return Data(**kwargs1), Data(**kwargs2)

    def __iter__(self):
        """Get an iterator for this data."""
        for i in range(len(self)):
            yield self[i]


class PreprocessedData(Data):
    """Preprocessed toxicity data."""

    def __init__(self, data, preprocessing):
        """Initialize by passing a Data object and a preprocessing function."""
        super().__init__(**data.kwargs)
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        """Get a preprocessed batch from this data."""
        batch_x, batch_y = super().__getitem__(idx)
        preprocessed_x = self.preprocessing(batch_x)
        return preprocessed_x, batch_y

    def split(self, fraction):
        """Split the data according to the given fraction."""
        data1, data2 = super().split(fraction)
        pre1 = PreprocessedData(data1, self.preprocessing)
        pre2 = PreprocessedData(data2, self.preprocessing)

        return pre1, pre2
