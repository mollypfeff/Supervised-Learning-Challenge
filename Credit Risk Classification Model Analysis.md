# Module 20 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
	* The purpose of this analysis was to assess whether borrowers are creditworthy or not by using an existing dataset of healthy and high-risk loans to create a logistic regression model that can predict whether a loan will be healthy or high-risk.
	
* Explain what financial information the data was on, and what you needed to predict.
	* The dataset contained the following information for each loan: loan size, interest rate, borrower income, number of accounts, derogatory marks, total debt, and loan status. The model predicts loan status, and uses the other variables as features to inform the prediction.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
	* In the original dataset, there are 75,036 healthy loans and 2,500 high-risk loans. 


* Describe the stages of the machine learning process you went through as part of this analysis.
	* Prepare the data
		* import the csv
		* load it into a dataframe
		* split the data into labels (loan_status) and features (all other columns)
	* Use train_test_split function to split labels and features into a group to train the model and a group to test the model.
	* Logistic Regression Model
		* Import the LogisticRegression model from SKLearn, which is the machine learning model we used for both models.
		* Instantiate the model
		* fit the model using training data
		* Use the model to make predictions using the testing portion of the features data
	* Determine the performance/efficacy of the model using multiple metrics, including:
		* balanced accuracy score
		* confusion matrix
		* classification report


* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
	* We used LogisticRegression in both prediction models. For Model 1 we used the original set of data, but for Model 2, we ran RandomOverSampler to resample the training data.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: Logistic Regression (with original data)
	* Balanced accuracy: 96.8%
	* Precision: 100% for healthy loans, 84% for high-risk loans
	* Recall: 99% for healthy loans, 94% for high-risk loans
 
* Machine Learning Model 2: Logistic Regression (with oversampled data)
	  * Balanced accuracy: 99.4%
	  * Precision: 100% for healthy loans, 84% for high-risk loans
	  * Recall: 99% for both healthy and high-risk loans

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
	* Model 2 (Logistic Regression with Oversampled Data) appears to perform best, due to it having a higher % accuracy and a higher % recall of high-risk loans than Model 1.

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
	* It is important to scrutinize the performance of the model when predicting 1's, which signify loans with a high risk of defaulting. If the purpose of the model is to identify the creditworthiness of borrowers, the model should be able to identify high-risk loans as accurately as possible.
