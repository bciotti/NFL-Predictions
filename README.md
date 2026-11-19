# NFL Over/Under Prediction with Neural Networks (Wolfram Language)

Capstone project for the Wolfram U Neural Networks Level 2 certification.

## Question
Can a neural network predict whether an NFL game's total points will go over or under the bookmaker's line?

## Data
Kaggle "NFL scores and betting data" (14,358 games; about 8,300 train / 2,000 test after removing rows with missing values). Link: https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data

## Method
Self-normalizing feed-forward regression network (adapted from a Wolfram reference architecture); tuned depth, width, dropout, L2 regularization, batch size. Compared to Wolfram's Predict[] models and a classification net.

## Results
| Model | Accuracy |
|---|---|
| Baseline (`[your baseline]`) | `[x]%` |
| Regression net | 54.0% |
| Classification net | 51.5% |

Sampling error is about ±1.1% on the test set.

## Limitations
Single random split; `[chronological-split result if you run it]`.

## Files
- `NFLProjectFeb12.nb`: the notebook
