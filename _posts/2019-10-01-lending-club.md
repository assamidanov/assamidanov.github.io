---
title: "Lending Club Loan Data: Analyze Lending Club's issued loans"
date: 2019-10-01
tags: [Data science]
#header:
  #image: "/images/Arqaam/idps1.jpg"
excerpt: "Machine Learning"
---

## Question: Can we build a machine learning model that can accurately predict if a borrower will pay off their loan on time or not?

### Introduction and Problem Background

<img src="{{ site.url }}{{ site.baseurl }}/images/lending/lend.png" alt="linearly separable data">

Credit modeling is a well known data science problem that focuses on modeling a borrower's credit risk. Credit has played a key role in the economy for centuries and some form of credit has existed since the beginning of commerce. We'll be working with financial lending data from Lending Club. Lending Club is a marketplace for personal loans that matches borrowers who are seeking a loan with investors looking to lend money and make a return.

Each borrower fills out a comprehensive application, providing their past financial history, the reason for the loan, and more. Lending Club evaluates each borrower's credit score using past historical data and assign an interest rate to the borrower. The interest rate is the percent in addition to the requested loan amount the borrower has to pay back. Lending Club also tries to verify each piece of information the borrower provides but it can't always verify all of the information (usually for regulation reasons).

A higher interest rate means that the borrower is riskier and more unlikely to pay back the loan while a lower interest rate means that the borrower has a good credit history is more likely to pay back the loan. The interest rates range from 5.32% all the way to 30.99% and each borrower is given a grade according to the interest rate they were assigned. If the borrower accepts the interest rate, then the loan is listed on the Lending Club marketplace.

Investors are primarily interested in receiving a return on their investments. Approved loans are listed on the Lending Club website, where qualified investors can browse recently approved loans, the borrower's credit score, the purpose for the loan, and other information from the application. Once they're ready to back a loan, they select the amount of money they want to fund. Once a loan's requested amount is fully funded, the borrower receives the money they requested minus the origination fee that Lending Club charges.

This blog is dedicated to build machine learning models on credit records. By using the Lending Club website‘s data, three different machine learning models were implemented to be able to forecast the probability of borrower’s ability to pay the loan on time or not. Prior performing machine learning model, it is necessary to clean the data and prepare the features.

### Data Cleaning

After analyzing each column, it was concluded that the below features need to be removed. The information isn't available to an investor before the loan is fully funded and it will not be included in the model.

```python
  drop_columns = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 'sub_grade', 'emp_title', 'issue_d']
  loans = loans_2007.drop(drop_columns, axis=1)
  loans = loans.drop(['zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp'], axis=1)
```

All of these below columns leak data from the future, meaning that they're describing aspects of the loan after it's already been fully funded and started to be paid off by the borrower.

```python
loans = loans.drop(['total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt'], axis =1)
```

'Loan_status' was chosen as a target column, since it's the only column that directly describes if a loan was paid off on time, had delayed payments, or was defaulted on the borrower. Currently, this column contains text values and we will need to convert it to a numerical one for training a model.

<img src="{{ site.url }}{{ site.baseurl }}/images/lending/loan_status.png" alt="linearly separable data">

We remove all the loans that don't contain either "Fully Paid" and "Charged Off" as the loan's status and then transform the "Fully Paid" values to 1 for the positive case and the "Charged Off" values to 0 for the negative case. We also  look for any columns that contain only one unique value and remove them. These columns won't be useful for the model since they don't add any information to each loan application.

```python
  status_dict ={
    'loan_status': {
    'Fully Paid': 1,
    'Charged Off': 0
    }
    }
  loans = loans.replace(status_dict)
  loans = loans[(loans['loan_status']==1) | (loans['loan_status']==0)]

  drop_columns = []
  for i in loans.columns:
    col_series = loans[i].dropna()
    if len(col_series.unique()) == 1:
      drop_columns.append(i)
  loans = loans.drop(drop_columns, axis=1)
```

### Feature Engineering

We'll prepare the data for machine learning by focusing on handling missing values, converting categorical columns to numeric columns, and removing any other extraneous columns we encounter throughout this process.

This is because the mathematics underlying most machine learning models assumes that the data is numerical and contains no missing values. To reinforce this requirement, *scikit-learn* will return an error if you try to train a model using data that contain missing values or non-numeric values when working with models like linear regression and logistic regression.

Let's use the strategy of removing the pub_rec_bankruptcies column first, since nearly 94% of values are in the same  category. Then, we removed all rows containing any missing values at all to cover both of these cases. This way, we only remove the rows containing missing values for the "emp_length", title and revol_util columns, but not the  pub_rec_bankruptcies column.

```python
loans = loans.drop('pub_rec_bankruptcies', axis=1)
loans = loans.dropna()
```

While the numerical columns can be used natively with scikit-learn, the object columns that contain text need to be converted to numerical data types. Let's return a new Dataframe containing just the object columns so we can explore them in more depth.We erred on the side of being conservative with the 10+ years, < 1 year and n/a mappings. We assume that people who may have been working more than 10 years have only really worked for 10 years. We also assume that people who've worked less than a year or if the information is not available that they've worked for 0. This is a general heuristic but it's not perfect.

```python
loans['int_rate'] = loans['int_rate'].str.replace('%',"").astype("float")
loans["revol_util"] = loans["revol_util"].str.replace('%',"").astype("float")
loans['loan_status'] = loans['loan_status'].astype("float")

mapping = {'emp_length' : {
   "10+ years": 10,
    "< 1 year": 0,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "1 year": 1,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "n/a": 0
}}

loans = loans.replace(mapping)
loans['emp_length'] = loans['emp_length'].astype("float")
loans = loans.drop(['last_credit_pull_d', 'addr_state', 'title', 'earliest_cr_line'], axis=1)
```

Let's now encode the "home_ownership", "verification_status", "purpose", and term columns as dummy variables so we can use them in our model.

```python
loans = pd.get_dummies(loans, columns =['home_ownership', 'verification_status', 'purpose', 'term'])
```

### Making predictions

We noticed that there's a class imbalance in our target column, "loan_status". There are about 6 times as many loans that were paid off on time (positive case, label of 1) than those that weren't (negative case, label of 0). Imbalances can cause issues with many machine learning algorithms, where they appear to have high accuracy, but actually aren't learning from the training data. Because of its potential to cause issues, we need to keep the class imbalance in mind as we build machine learning models.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

cols = list(loans.columns)
cols.remove("loan_status")
features = loans[cols]
target = loans['loan_status']
predictions = cross_val_predict(lr, features, target, cv=3)
predictions = pd.Series(predictions)
# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])
# Rates
tpr = tp  / (tp + fn)
fpr = fp  / (fp + tn)
```

As you can see from the last screen, our fpr and tpr are around what we'd expect if the model was predicting all ones.We can do this by setting the class_weight parameter to "balanced" when creating the LogisticRegression instance. This tells scikit-learn to penalize the misclassification of the minority class during the training process. The penalty means that the logistic regression classifier pays more attention to correctly classifying rows where "loan_status" is 0. This lowers accuracy when loan_status is 1, but raises accuracy when loan_status is 0.

By setting the class_weight parameter to balanced, the penalty is set to be inversely proportional to the class frequencies. This would mean that for the classifier, correctly classifying a row where loan_status is 0 is 6 times more important than correctly classifying a row where loan_status is 1.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
lr = LogisticRegression(class_weight='balanced')
predictions = cross_val_predict(lr, features, target, cv=3)
predictions = pd.Series(predictions)
```

We significantly improved false positive rate in the last screen by balancing the classes, which reduced true positive rate. Our true positive rate is now around 63%, and our false positive rate is around 61%. From a conservative investor's standpoint, it's reassuring that the false positive rate is lower because it means that we'll be able to do a better job at avoiding bad loans than if we funded everything. However, we'd only ever decide to fund 63% of the total loans (true positive rate), so we'd immediately reject a good amount of loans.

We can also specify a penalty manually if we want to adjust the rates more. To do this, we need to pass in a dictionary of penalty values to the class_weight parameter.

```python
penalty = {
    0: 10,
    1: 1
    }
lr = LogisticRegression(class_weight=penalty)

predictions = cross_val_predict(lr, features, target, cv=3)
predictions = pd.Series(predictions)
```

It looks like assigning manual penalties lowered the false positive rate to 22.5%, and thus lowered our risk. Note that this comes at the expense of true positive rate. While we have fewer false positives, we're also missing opportunities to fund more loans and potentially make more money. Given that we're approaching this as a conservative investor, this strategy makes sense, but it's worth keeping in mind the tradeoffs.

Let's try a more complex algorithm, random forest. Random forests are able to work with nonlinear data, and learn
complex conditionals.  Training a random forest algorithm may enable us to get more accuracy due to columns that correlate nonlinearly with loan_status.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight="balanced", random_state=1)
predictions = cross_val_predict(rf, features, target, cv=3)
predictions = pd.Series(predictions)
```

### Conclusion

Unfortunately, using a random forest classifier didn't improve our false positive rate. The model is likely weighting too heavily on the 1 class, and still mostly predicting 1s. We could fix this by applying a harsher penalty for misclassifications of 0s.
Ultimately, our best model had a false positive rate of 22.5%, and a true positive rate of 22.8%. For a conservative investor, this means that they make money as long as the interest rate is high enough to offset the losses from 22.5% of borrowers defaulting, and that the pool of 22.8% of borrowers is large enough to make enough interest money to offset the losses.
