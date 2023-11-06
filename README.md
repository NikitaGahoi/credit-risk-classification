# Credit-risk-classification
## Overview of the Analysis
This comprehensive analysis was focused on addressing the critical task of credit risk classification by leveraging machine learning models. The overarching objective was to create and evaluate models capable of determining the creditworthiness of borrowers using a dataset containing historical lending activity from a peer-to-peer lending services company. The primary aim was to predict loan status, with '0' representing healthy loans and '1' indicating high-risk loans. This analysis entailed the utilization of various financial features, including loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt.

The analysis unfolded through several pivotal stages:

- **Data Preprocessing:** The loan status labels were stored in the 'y' variable. Features, represented by all columns except 'loan_status,' were stored in the 'X' variable. The balance of labels was checked using 'value_counts,' revealing 75,036 healthy loans and 2,500 high-risk loans in the dataset.

- **Data Splitting:** The data was divided into training and testing sets using the 'train_test_split' module from sklearn. This created four variables: 'X_train,' 'X_test,' 'y_train,' and 'y_test.' A 'random_state' of 1 was set to ensure consistent splits, providing the same data points for both training and testing across multiple runs.

- **Model Creation (Logistic Regression) with Original Data:** A Logistic Regression model was constructed using 'LogisticRegression' from sklearn, incorporating a 'random_state' of 1. The model was fitted with the training data ('X_train' and 'y_train') and employed for predictions on the testing data labels. Accuracy score was calculated using 'balanced_accuracy_score' from sklearn, utilizing 'y_test' and the predictions.

- **Confusion Matrix & Classification Report:** A confusion matrix and classification report was generated for the model using 'confusion_matrix' and 'classification_report' respectively from sklearn, based on 'y_test' and the model's predictions.
  
- **Data Resampling (RandomOverSampler):** To cope with the class imbalance in the dataset, 'RandomOverSampler' from imbalanced-learn was utilized to address class imbalance. The model was fitted with the training data ('X_train' and 'y_train'). Resampled data, denoted as 'X_ros_model' and 'X_ros_model,' was generated. The number of distinct values in the resampled labels data was obtained using 'Counter'. The resampled data was divided into training and test set using the 'train_test_split' module from sklearn.

- **Model Creation (Logistic Regression) with Resampled Data:** A Logistic Regression model was created using the resampled data. The model was fitted with the resampled data and used for predictions. Accuracy score, confusion matrix, and classification report were obtained for the resampled model.

## Results
**Machine Learning Model 1: Logistic Regression**
The accuracy predicted for the model is 0.99, which is a great accuracy. The model did great for low-risk loans(0){precision = 1, recall = 0.99}, however, when you look at the precision and recall score to high-risk loans(1), the values are 0.85 and 0.91 respectively.

**Reason:** The data is imbalanced as 96.77% of the target values (75036 out of 77536) are for the healthy loans/low-risk loans(0). Therefore the model trained well for the  healthy loans  whereas the data for high-risk loans was insufficient to train the model accurately to predict the defaulters/high-risk loans

<img width="500" alt="image" src="https://github.com/NikitaGahoi/credit-risk-classification/assets/136101293/2318a83e-6ee1-40c9-a7a9-ed6a8373fc09">


**Machine Learning Model 2: RandomOverSampler**
This model did a great job in predicting both the healthy and the high-risk loans as can be inferred from the high balanced accuracy score of 99.50%. This model has a precision score of 100% for the healthy loans and 99% for the high-risk loans. The precision scores imply that the healthy loans were classified correctly as positive 100% of the times. However, for the high-risk loans, the classification was correct only 84% of the times.

This model has a recall score of 99% for the healthy loans and 100% for the high-risk loans. The scores imply that for all the instances where the loans were actually healthy or when they were high-risk, 99% of the times they were classified correctly.

<img width="500" alt="image" src="https://github.com/NikitaGahoi/credit-risk-classification/assets/136101293/52a49c75-e087-4e94-ad1e-03f177ff736f">


## Summary

In summary, the analysis reveals that the RandomOverSampler model (Machine Learning Model 2) surpasses the performance of the Logistic Regression model in terms of accuracy, balanced accuracy, precision, and recall. The second model also proves to be highly effective in addressing the class imbalance issue present in the dataset.

Notably, the RandomOverSampler model excels in the crucial task of identifying high-risk loans, which is of paramount significance for lending institutions. Reducing the probability of defaulting loans by accurately assessing creditworthiness has a substantial impact on the overall profitability of a lending service company.

Hence, we strongly recommend employing the RandomOverSampler method to address class imbalance by resampling the data, followed by the implementation of Logistic Regression. This approach guarantees superior accuracy and recall scores, making it the optimal choice for a lending service company when evaluating and managing credit risk.

