# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a supervised machine learning model to predict whether an individual’s income exceeds $50K per year based on demographic and employment-related features. The model is a Random Forest classifier, implemented using the scikit-learn library.

The model takes both categorical and numerical features as input. Categorical variables are transformed using one-hot encoding, and the target variable is binarized into two classes: <=50K and >50K. The trained model is serialized and deployed through a FastAPI application for inference.

## Intended Use

This model is only intended for educational purposes, and demonstrates how to train, evaluate, and deploy a model using tools such as FastAPI and CI/CD workflows.

## Training Data

The model is trained on the UCI Census Income dataset, which contains demographic and employment-related information such as age, education, occupation, and hours worked per week. The training data was sliced to 25% of the original size of the dataset.

## Evaluation Data

The dataset is split into training and testing subsets. The model is evaluated on a held-out test set that was not used during training.

## Metrics

On the test dataset, the model achieved:

Precision: 0.7231
Recall: 0.6229
F1 Score: 0.6693

## Ethical Considerations

This model is trained on census data that includes demographic attributes and thus may learn and reinforce societal biases present in the data. 

## Caveats and Recommendations

This model is trained on historical data, which likely does not reflect the current economic conditions.