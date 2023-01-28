# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- who create the model: Sejong Heo
- type of model: 'RandomForestClassifier' from 'sklearn' 
- hyper parameter:
  - random_state: 42
  - n_jobs: -1

## Intended Use

This model predicts the salary of a person into 2 classes '<=50K' and  '>50K', based on the census data especially, considering following categorical features:

- "workclass"
- "education"
- "marital-status"
- "occupation"
- "relationship"
- "race"
- "sex"
- "native-country"

## Training Data

[census dataset](https://archive.ics.uci.edu/ml/datasets/census+income) is used for training. Only 80% of the data are used for trainging.

## Evaluation Data

[census dataset](https://archive.ics.uci.edu/ml/datasets/census+income) is used for evaluation. Only 10% of the data are used for evaluation.

## Metrics

3 metrics were used for evaluating the model's performance: precision, recall, and fbeta. The model achieves the following result:

- precision: 0.7
- recall: 0.6
- fbeta: 0.7

## Ethical Considerations

The result of this classification should not be used for discriminate the people based on race, gender, etc. The result only shows the phenomenon of current situation when the dataset was gathered.

## Caveats and Recommendations

The census dataset was gathered from the USA in 1994. Thus, this model will not fit to the current USA and other areas of the world.