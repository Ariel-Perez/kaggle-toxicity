#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains the base model class."""
import uuid
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from keras.models import load_model
from data import PreprocessedData


class Model:
    """Keras model wrapper."""

    INTERNAL_VARIABLES = []

    def fit_preprocessor(self, data):
        """Do any preparation for the preprocessing to happen."""
        pass

    def preprocess(self, data):
        """Do any preprocessing required for the features."""
        return data

    def predict(self, data, batch_size=128):
        """Get predictions from the given data."""
        inputs = self.preprocess(data)
        return self.model.predict(inputs, batch_size=batch_size)

    def unique_checkpoint_file_name(self):
        """Get a unique file name to save checkpoints."""
        values = {
            "name": type(self).__name__,
            "id": str(uuid.uuid4())
        }
        prefix = "models/checkpoints/{name}.{id}".format(**values)
        suffix = "{epoch:02d}.hdf5"
        return '.'.join([prefix, suffix])

    def roc(self, validation):
        """Compute the Area Under Curve - ROC metric."""
        values = ((batch_y.as_matrix(), self.predict(batch_x))
                  for batch_x, batch_y in validation)
        y, p = zip(*values)

        y = np.concatenate(y)
        p = np.concatenate(p)

        return roc_auc_score(y, p)

    def train(self, training, validation,
              epochs, early_stop=True):
        """Train the model."""
        self.fit_preprocessor(training)

        preprocessed_training = PreprocessedData(training, self.preprocess)
        preprocessed_validation = PreprocessedData(validation, self.preprocess)

        def compute_and_print_roc(epoch, logs):
            roc = self.roc(validation)
            print('\rval_roc: %f%s\n' % (roc, ' ' * 80))

        callbacks = [
            ModelCheckpoint(
                self.unique_checkpoint_file_name(), verbose=1, period=5),
            LambdaCallback(on_epoch_end=compute_and_print_roc)
        ]

        if early_stop:
            callbacks.append(EarlyStopping(patience=2))

        self.model.fit_generator(
            preprocessed_training,
            epochs=epochs,
            steps_per_epoch=len(preprocessed_training),
            validation_data=preprocessed_validation,
            validation_steps=len(preprocessed_validation),
            # use_multiprocessing=True,
            callbacks=callbacks
        )

    def save(self, model_path, additional_path=None):
        """Save the current model."""
        self.model.save(model_path)

        if self.INTERNAL_VARIABLES and additional_path is None:
            raise ValueError(
                "Need the additional_path argument "
                "to save additional information")

        if self.INTERNAL_VARIABLES:
            attrs = {key: getattr(self, key)
                     for key in self.INTERNAL_VARIABLES}

            with open(additional_path, 'wb') as handle:
                pickle.dump(attrs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, model_path, additional_path=None):
        """Load the model from the given path."""
        model = cls.__new__(cls)
        model.model = load_model(model_path)

        if cls.INTERNAL_VARIABLES and additional_path is None:
            raise ValueError(
                "Need the additional_path argument "
                "to load additional information")

        if cls.INTERNAL_VARIABLES:
            with open(additional_path, 'rb') as handle:
                attrs = pickle.load(handle)

            for key in cls.INTERNAL_VARIABLES:
                setattr(model, key, attrs[key])

        return model
