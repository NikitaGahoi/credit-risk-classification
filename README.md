# Credit-risk-classification
## Overview of the Analysis
This comprehensive analysis was focused on addressing the critical task of credit risk classification by leveraging machine learning models. The overarching objective was to create and evaluate models capable of determining the creditworthiness of borrowers using a dataset containing historical lending activity from a peer-to-peer lending services company. The primary aim was to predict loan status, with '0' representing healthy loans and '1' indicating high-risk loans. This analysis entailed the utilization of various financial features, including loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt.

The analysis unfolded through several pivotal stages:

- **Data Preprocessing:** The loan status labels were stored in the 'y' variable. Features, represented by all columns except 'loan_status,' were stored in the 'X' variable. The balance of labels was checked using 'value_counts,' revealing 75,036 healthy loans and 2,500 high-risk loans in the dataset.

- Data Splitting: The data was divided into training and testing sets using the 'train_test_split' module from sklearn. This created four variables: 'X_train,' 'X_test,' 'y_train,' and 'y_test.' A 'random_state' of 1 was set to ensure consistent splits, providing the same data points for both training and testing across multiple runs.

- Model Creation (Logistic Regression) with Original Data:A Logistic Regression model was constructed using 'LogisticRegression' from sklearn, incorporating a 'random_state' of 1. The model was fitted with the training data ('X_train' and 'y_train') and employed for predictions on the testing data labels. Accuracy score was calculated using 'balanced_accuracy_score' from sklearn, utilizing 'y_test' and the predictions.

- Confusion Matrix & Classification Report: A confusion matrix and classification report was generated for the model using 'confusion_matrix' and 'classification_report' respectively from sklearn, based on 'y_test' and the model's predictions.
  
- Data Resampling (RandomOverSampler): To cope with the class imbalance in the dataset, 'RandomOverSampler' from imbalanced-learn was utilized to address class imbalance. The model was fitted with the training data ('X_train' and 'y_train'). Resampled data, denoted as 'X_ros_model' and 'X_ros_model,' was generated. The number of distinct values in the resampled labels data was obtained using 'Counter'. The resampled data was divided into training and test set using the 'train_test_split' module from sklearn.

- Model Creation (Logistic Regression) with Resampled Data: A Logistic Regression model was created using the resampled data. The model was fitted with the resampled data and used for predictions. Accuracy score, confusion matrix, and classification report were obtained for the resampled model.
