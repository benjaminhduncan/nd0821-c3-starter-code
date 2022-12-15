# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Ben Duncan created the model. It is a GradientBoostingClassifier implemented in scikit-learn version 1.2.0. The model utilizes Grid Search hyperparameter tune, varying 'n_estimators', 'max_depth', and 'learning_rate'.

## Intended Use
This model should be used to predict the expected income of an individual based on a handfull of census attributes. The users are individuals wanting to predict income level classification based on census attributes.

## Training Data
The data used originates from the UCI Machine Learning Repository Census Income Data Set. Located at https://archive.ics.uci.edu/ml/datasets/census+income. Extraction was done by Barry Becker from the 1994 Census database. The target class for this set is a binary salary estimation (>50K, <=50K). 

The set was split into a train and test set using a 80-20 split. No stratification or k-fold methods were used. For training the machine learning model, a OneHotEncoder and LabelBinarizer preprocessing step was applied to the data.

## Evaluation Data
The test set from the test/train split was used in evaluating the model.

## Metrics
The model was evaluated against three metrics: fbeta, precision, recall as implemented in the scikit-learn scorers in version 1.2.0.

The performance metrics for this trained model are:
fbeta: 0.7018612521150592
precision: 0.7779444861215303
recall: 0.6393341553637485

## Ethical Considerations
Given the nature of predicting an individuals salary based on a limited number of features from a census data set, special considerations should be taken when using the model. There are many factors which contribute to an individuals salary and the predictions made from this model should be used with care.

## Caveats and Recommendations
The training data set used contains census information including race, nationality, sex and several other personal factors. Due to limited samples from many classes there is potentially unfairness and bias present in this data set. Additionally, this model's classification performance in all three metrics should be considered when using it. Additional sample, bias studies and model refinement should be considered.