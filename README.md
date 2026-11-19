# NFL Over/Under Prediction with Neural Networks (Wolfram Language)

Capstone project for the Wolfram U Neural Networks Boot Camp, Level 2 certification.

## Question
Can a neural network predict whether the combined point total of an NFL game will go under the bookmaker's over/under line?

## Data
Kaggle dataset ["NFL scores and betting data"](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data): 14,358 games from the 1960s onward, with scores, point spreads, over/under lines, dates, venues, and weather. After dropping two sparsely populated columns and removing rows with missing values, about 10,500 games remained. About 8,300 were used for training and the rest were held out for testing (a random split of roughly 80/20).

## Target accuracy
At standard -110 betting odds, a bettor must win more than about 52.4% of over/under bets to profit. Assuming roughly 1.85% of bets push, this falls to about **51.4%**, which I used as the target accuracy.

## Method
- Numeric encoding of the inputs with Wolfram's `FeatureExtraction`
- Feed-forward regression network in the Wolfram neural network framework (self-normalizing style, adapted from a Wolfram Neural Net Repository example), with AlphaDropout and L2 regularization
- Hyperparameters (dropout, depth, width, regularization, stopping criterion, batch size) tuned one at a time
- Compared against Wolfram's built-in `Predict[]` models
- Also tried a classification network and averaging ensembles of several trained nets

## Results (single random train/test split)
| Model | Test accuracy |
|---|---|
| Break-even threshold at -110 odds (with pushes) | 51.4% |
| Regression network | 54.0% |
| Classification network | 51.5% |

With a test set of roughly 2,000 games, accuracy has a sampling error of about ±1.1 percentage points. On this split the regression network had lower RMSE and higher accuracy than `Predict[]`'s neural network, and beat about half of the other `Predict[]` methods on RMSE. Tree-based methods were competitive, which I did not fully explain. As an illustration only, the nets were also run on one recent Super Bowl, where most predicted the under (a single game, not evidence).

## Limitations
- One random split and one seed; results vary by randomization
- Random splitting mixes seasons from different eras, so the model can see "future" games. A chronological split would be more realistic
- No always-under or other naive baseline was computed in this notebook
- Team names were mapped to a single ID across franchise relocations, which was not compared against skipping that step

## Possible next steps
- Chronological train/test split; repeated resampling for a better accuracy estimate
- Encode week of season as "percentage through the season" (a suggestion from my instructor)
- Investigate why decision trees perform so well here

## Files
- `NFLProjectFeb12.nb`: the Wolfram notebook
