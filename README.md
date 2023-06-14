# MLB Game Predictions

Machine learning that predicts the outcomes of any MLB game. Data are from 2018 - 2022 seasons. 
Current accuracy on test data:
- mean running average of 2 and 3 games have a test accuracy of 70%
- median running average of 2 games has a test accuracy of 66.6%

## Usage

```python
python mlb_ml_classify_deep_learn.py tune or python mlb_ml_classify_deep_learn.py notune
#check accuracy of running averages
python mlb_ml_classify_deep_learn.py test
```

```bash
Columns dropped  >= 0.90: ['RBI', 'onbase_plus_slugging', 'ER', 'strikes_total']
Number of samples: 21234

### Current prediction accuracies - Deep Neural Network
Train Accuracy: 0.9793
Train Loss: 0.9655
Validation Accuracy: 0.05
Validation Loss: 0.09
```

## Running average outcomes
![](https://github.com/bszek213/ml_mlb/blob/main/best_mean_ma.png)
![](https://github.com/bszek213/ml_mlb/blob/main/best_median_ma.png)
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
