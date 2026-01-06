# Titanic Survivor Prediction

This repository contains a complete machine learning project to predict passenger survival on the Titanic dataset.  
The focus of this project is clarity reproducibility and strong fundamentals rather than leaderboard chasing.

The workflow follows standard data science practices and is suitable for learning interviews and portfolio use.

## Project Description

The Titanic dataset is a binary classification problem where the objective is to predict whether a passenger survived the disaster based on available passenger information.

This project demonstrates how to build an end to end machine learning solution starting from raw data and ending with a saved trained model ready for inference.

## Dataset Description

The dataset includes passenger level information such as

- Passenger class
- Sex
- Age
- Fare
- Family size
- Port of embarkation

The target variable is Survived.

PassengerId is treated strictly as an identifier and excluded from modeling to prevent data leakage.

## Approach Overview

The modeling workflow follows these steps

1 Data preprocessing and encoding without pipelines  
2 Train test split  
3 Training multiple baseline models  
4 Model comparison using accuracy and F1 score  
5 Hyperparameter tuning with cross validation  
6 Feature selection using Random Forest feature importance  
7 Final model training and validation  
8 Model export for future use  

## Models Evaluated

The following machine learning models were trained and evaluated

- Logistic Regression
- K Nearest Neighbors
- Support Vector Machine
- Random Forest
- Gradient Boosting

Random Forest was selected as the final model due to its balance between performance interpretability and stability.

## Final Model Summary

The final model is a tuned Random Forest classifier trained on selected high importance features.

Key characteristics

- Hyperparameters optimized using GridSearchCV
- Five fold cross validation
- Optimization metric F1 score
- Balanced class weights to handle class imbalance
- Reduced feature set to minimize noise

Performance summary

- Test accuracy approximately 0.81
- Test F1 score approximately 0.76
- Cross validation performance consistent with test results

## Feature Selection

Feature importance from the trained Random Forest model was used to identify the most relevant predictors.

Final selected features include

- Sex
- Fare
- Passenger class
- Age
- Family size
- Embarked

PassengerId was removed explicitly to avoid leakage.

## Model Export

The trained model and feature list are saved using joblib for reuse.

Exported files

- titanic_random_forest_model.pkl
- titanic_features.pkl

These files can be loaded later to perform predictions on new data.

## How to Run

1 Clone the repository  
2 Install required dependencies  
3 Run the training script or notebook  
4 Use the saved model files for inference  

Required libraries include

- pandas
- numpy
- scikit learn
- joblib

## Key Takeaways

This project highlights

- The importance of F1 score for imbalanced classification
- The value of cross validation for generalization
- Why identifier features must be removed
- How feature selection improves model stability
- How to build an end to end ML solution without pipelines

## Future Improvements

Potential enhancements include

- Additional feature engineering
- Gradient boosting and advanced ensemble methods
- Probability calibration
- Model deployment
- Kaggle score optimization

