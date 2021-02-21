# Home Credit Default Risk

<img src="https://github.com/HebaAladdin/Kaggle-home-credit-risk/blob/main/loan.jpg" alt="loan" width="600" height="400"/>

## Project Description
In this series of notebooks, we will take a look at the ["Home credit default risk"](https://www.kaggle.com/c/home-credit-default-risk#description) machine learning competition currently hosted on Kaggle. The objective of this competition is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. This is a standard supervised classification task:

* **Supervised**: The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features

* **Classification**: The label is a binary variable, 0 (will repay loan on time), 1 (will have difficulty repaying loan)


### Datasets
* **application_{train|test}.csv**
Main table, broken into two files for Train (with TARGET) and Test (without TARGET).
* **bureau.csv**
All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
* **bureau_balance.csv**
Monthly balances of previous credits in Credit Bureau.
* **POS_CASH_balance.csv**
Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
* **credit_card_balance.csv**
Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
* **previous_application.csv**
All previous applications for Home Credit loans of clients who have loans in our sample.
* **installments_payments.csv**
Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.

### Dependencies

```
Python 3.7 
sklearn
lightgbm
Pandas
Numpy
```

For visualization

```
matplotlib
seaborn
```

### Running

1- Exploratory data analysis(EDA)

This notebook is for getting business insights from the dataset through EDA. Navigate to the the folder "EDA" and run the following notebook
```
Exploratory Data Analysis.ipynb
```

2- Data preparation

This notebook is for preparing the dataset with feature engineering and feature selection techniques to feed it later on to the machine learning models. Navigate to the folder "Modeling" and run the following notebook
```
Data Preparation.ipynb
```

3- Models

This collection of notebooks aim to run three machine learning models(Logistic regression, Random forest, Light Gradient Boosting Machine) on 4 different datasets. Navigate to the folder "Modeling" and run the following notebooks 

   - Raw application dataset without manual features

```
Modeling - Raw Dataset.ipynb
```
   - Domain knowladge features were added to the raw dataset
   
```
Modeling - Domain Dataset.ipynb
```
   - Aggregated features from bureau and bureau balance were added to the raw dataset
   
```
Modeling - Aggregated Dataset v1.ipynb
Modeling - Aggregated Dataset v2.ipynb
```


### Feature Engineering

#### Features based on my intuition
Based on my research about the topic as well as brainstorming lead me to think about handful of features that can be relative to our model based on the data sources provided in the competition. 

* Days since customer had payment past due
* Ratio between credit usage and credit limit
* Number of credit cards the applicants hold
* Number of active loans reported by Credit bureau
* Number of closed loans reported by Credit bureau
* Loan to income ratio
* Loan length payment (duration of the loan)
* Amount left to pay at the time of current application
* Difference between actual and expected last payment date
* Max credit limit the applicants got approved for on previous applications

Note: I ended up not implementing all of them only the features that can be calculated from bureau and bureau_balance data sources for future work we can dig deeper into the other data sources.

#### Features based on domain knowladge
Researching about the problem in hand I found this article from Wells Fargo explains what factors are looked at while providing money to borrowers. https://www.wellsfargo.com/financial-education/credit-management/five-c/

I am also familiar with the concept of Iscore (Egyptian Credit Bureau) here in Egypt as you get assessed by your previous payment history and credit history. Most of the lenders here in Egypt relay on this metric as a guidance when giving loans to applicants. https://www.i-score.com.eg/en/information-for-individuals/home/

Similar to Iscore is the "FICO® scores" which inspired the following features to include in our training set:

- Debt-to-income ratio(DIR) = Credit amount of the loan / Total Income = AMT_CREDIT/AMT_INCOME_TOTAL
- Annuity-to-income ratio(AIR) = Loan annuity / Total Income = AMT_ANNUITY/AMT_INCOME_TOTAL
- Annuity-to-credit ratio(ACR) = Loan annuity/ Credit amount of the loan = AMT_ANNUITY/AMT_CREDIT
- Days-employed-to-age ratio(DAR) = Number of days employed/ Age of applicant = DAYS_EMPLOYED/DAYS_BIRTH


#### Statistics computed by grouping by accounts and months
The supplementary tables (previous application, bureau records, installment etc) cannot be directly merged to the main table, because clients have various number of previous loans and different length of credit history. Thus a lot of statistical features can be computed by first grouping by current application ID and averaging/summing over both different account and records of different months. Some statistics I computed includes:
* mean, sum, max, median, variance ...
* the above statistic functions calculated on subset of accounts, such as all active accounts, approved/refused applications, the most recent application...

By adding those features I could get a better performing model than the baseline model with raw dataset features.

### Modeling
Model diversity are essential for later ensembling and comparison. I started with building a baseline model with logistic regression and a small regularization paramter (C= 0.00001) to avoid overfitting as I had a lot of features goining into training. The second model I tried was using random forest that gave me a slightly better performance than the baseline. Last trial was running a 10 K-fold training using gradient boosting decision trees using [LightGBM](https://lightgbm.readthedocs.io/en/latest/) and it proved to be outperforming on all the datasets I used.

Since we have an unbalanced dataset we will use a metric that can judge better with this types of problems it is a common classification metric known as the [Receiver Operating Characteristic Area Under the Curve (ROC AUC, also sometimes called AUROC)](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it)

### Results
The final results is based on running 10 kfold training using LightGBM model (we can use stratified folds while training as we have an unbalanced dataset)

|  Experiment | Train AUC  | Validation AUC  | 
| ------------ | ------------ | ------------ |
| Raw dataset  |  0.806430 | 0.758923 |
| Domain dataset  | 0.815190 | 0.766038  |
| Aggregated dataset v1  | 0.825520 | 0.766415  | 
|Aggregated dataset v2 | 0.815504 | 0.763560 | 


### Future work

- Expand the EDA more to cover different features and how they relate to each other.
- Work with other feature selection methods such as PCA to reduce the feature space to find significant features
- Add more features from the other datasources and check the feature importance graphs for a clue on what features were most important
- Use stratified folding technique while training to work with the unbalanced labels problem
- Experiment with other models and compare it to the LightGBM model we trained
- Try bagging, boosting and stacking models 
- Hyperparameter tunning to optimize the models that give better results
- Train on each data source separatly and use the prediction from one model as an input to the other model

### References 
- The [discussion board](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction) of the kaggle competition is a powerful source of information related to the current problem.
- This [article](https://towardsdatascience.com/a-machine-learning-approach-to-credit-risk-assessment-ba8eda1cd11f) on medium about building a credit risk model was also a nice source of information.
- An article for [factors to quantify credit risk](https://www.investopedia.com/ask/answers/022415/what-factors-are-taken-account-quantify-credit-risk.asp#:~:text=Several%20major%20variables%20are%20considered,macroeconomic%20considerations%2C%20such%20as%20economic) helped me to understand more about the defaulting problem effect on business
