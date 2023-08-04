# MLB Game Predictions

Machine learning that predicts the outcomes of any MLB game. Data are from 2014 - 2023 seasons. 
Current accuracy on test data:
- Teams that I have a >70% prediction probability: 'KCR', 'ATL'
- Regression DNN has a validation RMSE of 0.57 runs

## Usage

```python
python mlb_ml_classify_deep_learn.py tune or python mlb_ml_classify_deep_learn.py notune
#check accuracy of running averages
python mlb_ml_classify_deep_learn.py test
```

```bash
Correlated features   >= 0.90: ['RBI', 'onbase_plus_slugging', 'ER', 'strikes_total']
Number of samples: 62718

### Current prediction accuracies - DNN
Validation Accuracy: 1.0000
Train Accuracy: 1.0000
Validation Loss: 2.4500e-05
Train Loss: 2.4679e-06
```

## Running average outcomes
![](https://github.com/bszek213/ml_mlb/blob/main/best_mean_ma.png)
![](https://github.com/bszek213/ml_mlb/blob/main/best_median_ma.png)
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
