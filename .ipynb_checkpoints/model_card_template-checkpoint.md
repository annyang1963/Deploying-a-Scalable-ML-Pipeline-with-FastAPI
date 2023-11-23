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
The model's performance was evaluated using the Precision, Recall, and F-1 score as metrics.
These metrics were chosen because they provide comprehensive insights into a machine learning pipeline's performance. Precision measures the accuracy of positive predictions, F1 Score balances precision and recall, and Recall assesses the model's ability to capture actual positive instances. Precision is crucial when minimizing false positives is a priority, while Recall is essential for avoiding false negatives. The F1 Score offers a balanced evaluation when both precision and recall need consideration, allowing for a more comprehensive assessment of the model's effectiveness.  The models performance on these metrics is as follows:

Precision: 0.7164
Recall: 0.2458
F1-Score: 0.3661

## Ethical Considerations
Fairness/bias: The model may exhibit bias if the training data contains biases in its attributes. Care should be taken to assess and mitigate any bias in model predictions.

Privacy: The model uses personal data for prediction, however all personal data is completely anonymized and publicly available, so there are no major concerns on personal privacy.

Transparency: The model uses logistic regression, which is a transparent and interpretable model. Users can see and understand the factors that contribute to predictions.


## Caveats and Recommendations
The model's performance, especially in terms of recall, indicates that it may not capture all individuals with incomes greater than $50,000 per year.

Address any potential bias in the training data to ensure fair and equitable predictions.

Regularly update the model to account for changes in the data distribution and societal factors, and continuously monitor the model's performance and retrain it as needed to maintain accuracy and fairness.