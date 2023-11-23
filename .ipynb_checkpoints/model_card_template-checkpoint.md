# Model Card

## Model Details
Model Name: Census Income Prediction Model
Version: 1.0
Model Type: Binary Classification
Model Architecture: Logistic Regression
Created by: Erick Contreras
Date Created: 11/01/2023

## Intended Use
The Census Income Prediction Model is designed for predicting whether an individual's income exceeds $50,000 per year based on various demographic and employment-related details. The model is intended for use in scenarios where income prediction is used in decision-making, such as targeted marketing, financial planning, or social program eligibility assessment.


## Training Data
Dataset: Census Income Dataset
Data Preprocessing: The training data was preprocessed with one-hot encoding of categorical features and label binarization. The categorical features used for training include workclass, education, marital-status, occupation, relationship, race, sex, and native-country.


## Evaluation Data
Evaluation Dataset: Census Income Test Dataset
Data Preprocessing: The evaluation data was preprocessed using the same process as the training data, including one-hot encoding of categorical features and label binarization.


## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model's performance was evaluated using the following metrics on the evaluation data:

Precision: 0.8041
Recall: 0.5487
F1-Score: 0.6515
These metrics were chosen to assess the model's ability to correctly classify individuals with income greater than $50,000 per year (precision) and its ability to capture a substantial portion of such individuals (recall).


## Ethical Considerations
Fairness/bias: The model may exhibit bias if the training data contains biases in its attributes. Care should be taken to assess and mitigate any bias in model predictions.

Privacy: The model uses personal data for prediction, however all personal data is completely anonymized and publicly available, so there are no major concerns on personal privacy.

Transparency: The model uses logistic regression, which is a transparent and interpretable model. Users can see and understand the factors that contribute to predictions.


## Caveats and Recommendations
The model's performance, especially in terms of recall, indicates that it may not capture all individuals with incomes greater than $50,000 per year.

Address any potential bias in the training data to ensure fair and equitable predictions.

Regularly update the model to account for changes in the data distribution and societal factors, and continuously monitor the model's performance and retrain it as needed to maintain accuracy and fairness.