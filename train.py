#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the training script."""
import argparse
import pandas as pd
import data
import models


def build_model():
    """Build the model to be trained."""
    return models.CharMixed(vocab_size=100, max_len=128)


def train(model, data_path, epochs, model_path, extra_path=None):
    """Train a given model on the given data and save it."""
    labels=['toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate']

    df = data.Data(
        data=pd.read_csv(data_path),
        batch_size=128,
        features=["comment_text"],
        labels=labels)

    training, validation = df.split(0.8)

    model.train(training, validation, epochs, early_stop=True)
    model.save(model_path, extra_path)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='toxicity',
        description='Train a model for detecting toxicity in messages.'
    )

    parser.add_argument('--data', type=str, help='path to the training data', required=True)
    parser.add_argument('--save', type=str, help='path to save model file', required=True)
    parser.add_argument('--extra', type=str, help='path to save extra model parameters')
    parser.add_argument('--epochs', type=int, help='number of epochs to train for', required=True)

    args = parser.parse_args()
    model = build_model()
    train(model, args.data, args.epochs, args.save, args.extra)
