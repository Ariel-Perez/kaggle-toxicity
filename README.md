# Toxicity ML
Submission for the Toxic Comment Classification Challenge at [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### How to Use
Use the Jupyter notebooks provided:
- `training.ipynb`
- `submission.ipynb`

If running from the console, train with the following script:
```bash
train.py --data <path-to-train.csv> --save <model-path> --extra <model-parameters-path> --epochs <epochs>
```
To build a submission from an already trained model, use the following:
```bash
submit.py --data <path-to-test.csv> --model <model-path> --extra <model-parameters-path> --out <output-path>
```
