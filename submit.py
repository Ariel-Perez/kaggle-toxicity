#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the submit script."""
import argparse
import pandas as pd
import models


def load_model(model_path, extra_path):
    """Load the trained model."""
    model_class = models.CharMixed
    return model_class.load(model_path, extra_path)


def submit(model, data_path, out_path):
    """Train a given model on the given data and save it."""
    labels=['toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate']

    test_data = pd.read_csv(data_path)
    predictions = model.predict(test_data)

    submission = pd.DataFrame(predictions, columns=labels)
    submission.insert(0, "id", test_data.id)
    submission.to_csv(out_path, index=False)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='toxicity',
        description='Use a trained model for detecting toxicity in messages.'
    )

    parser.add_argument('--data', type=str, help='path to the test data', required=True)
    parser.add_argument('--model', type=str, help='path to load model file', required=True)
    parser.add_argument('--extra', type=str, help='path to load extra model parameters')
    parser.add_argument('--out', type=str, help='path to save the submission file', required=True)

    args = parser.parse_args()
    model = load_model(args.model, args.extra)
    submit(model, args.data, args.out)
