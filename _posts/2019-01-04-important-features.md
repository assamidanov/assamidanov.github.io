---
title: "Identifying Important Features Using Household Survey Data"
date: 2019-01-04
tags: [Data science]
#header:
  #image: "/images/Arqaam/idps1.jpg"
excerpt: "Machine Learning: Feature Importance"
---

## Background Information

Afghanistan faces one of the world’s most acute internal displacement crises; resulting of several factors such as protracted conflict, ongoing insecurity, and natural hazards. Displacement has become a familiar survival strategy for many Afghans and, in some cases, an inevitable part of life for two generations. As of 31 December 2018, Afghanistan has 2,598,000 total number of internally displaced persons (IDPs).

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/idps1.jpg" alt="linearly separable data">

Displacement affects all individuals differently with needs, vulnerabilities and protection risks evolving due to exhaustion of coping mechanisms and only basic emergency assistance provided following initial displacement. Inadequate shelter, food insecurity, insufficient access to sanitation and health facilities, as well as a lack of protection, often result in precarious living conditions that jeopardize the well-being and dignity of affected families.

## Aim of the Project

The survey was taken by Arqaam to monitor and evaluate the NGOs’ humanitarian assistance projects. It is also directed towards understanding the status-quo before the beginning of the project, and the impact the project had over time.
The survey has 237 comprehensive questions and 11260 respondents. The survey was taken in a randomly stratified approach. Jalalabad and Herat were chosen as a target region. It can be claimed with its variety of ethnic groups and displacement specifications both cities can be representative of the whole of Afghanistan.

This project aims to perform a machine-learning algorithm to predict the displacement status of the respondents based on the responses from the questionnaire. Then, to determine important features that define displacement status of residents by using state of art techniques.  I used two Machine Learning algorithms like Random Forest Classifier and CatBoost Classifier to identify these important features. I also used 4 feature importance techniques: Default feature importance, Permutation feature importance, Drop Column feature importance and Shapley Additive explanations (SHAP).By comparing the outcomes of these techniques, I will determine the most important and the least important features.

## Explanatory Data Analysis

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture1.png" alt="linearly separable data">

According to the bar chart above, 90.57% percent of the respondents are Host, which is 10198 people. 9.01% of the respondents displaced because of war and criminal violence. The other 0.28 and 0.14 displacements happened due to natural disasters and economic conditions.

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture2.png" alt="linearly separable data">

Since the project aimed to determine important features that constitute the status of the respondents. It is necessary to check correlation coefficients and remove one of the variables of the highly correlated variables.  The correlation map above shows that if the person owns the land it will do some activities on that land. Thus, we can remove land_activities which are more specific for our purpose.

## Machine Learning Model: Feature Importance

As per the problem statement, the emphasis is on identifying the importance of a particular feature in the machine learning model. This is a multiclass classification problem, wherein we need to identify the important features. It is also necessary to justify the reasons by showing why it was chosen those specific features and how it turns out to be important features. Identifying important features is very important due to the following reasons:

-	It will help to improve the models by concentrating on important features.
-	It will help to remove features that are not relevant, or which do not contribute to the model’s performance.

### Random Forest Model

Random Forest Classifier was chosen as a benchmark model since is it is often used for feature selection in a data science workflow. The reason is that the tree-based strategies used by random forests naturally rank by how well they improve the purity of the node. This means a decrease in impurity over all trees (called Gini impurity). Nodes with the greatest decrease in impurity happen at the start of the trees, while notes with the least decrease in impurity occur at the end of trees. Thus, by pruning trees below a particular node, we can create a subset of the most important features.

Because of the comprehensiveness of the data, and its survey specifications, which captures all the relevant information, the accuracy of the model is 95%. This result will create robustness in our important features test. However, I did not focus on tuning in hyperparameters to increase accuracy since it is not the primary goal of this project.

*Default feature importance*

The default feature importance technique (feature_importances_ in Scikit-Learn) is based on the concept of training a tree we can compute how much each feature contributes to decreasing the weighted impurity. In the case of Random Forest, it takes the average of the decrease in impurity over trees.  

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture3.png" alt="linearly separable data">

The bar chart above explains how much prediction changes if we change the features. As the variables are in factions, the overall summation will be 1.

It seems that the top 6 the most important features are:

-	Fam_origin – if the respondent’s family is originally from Jalalabad/Herat
-	Loc_herat – the region the respondent belongs to
-	Soc_assist - whether respondent’s family ever received any assistance from any organization or government, the local community or relatives
-	Income_monthly – monthly income
-	Food_exp - food expenditure
-	Hh_total - total household composition

Intuitively, the importance of these features makes sense to determine the status IDPs. However, this approach is assumed to be biased, as it tends to inflate the importance of continuous features or high-cardinality categorical variables. Hence, it is necessary to compare with other alternative approaches so that our results would be more robust.

*Permutation feature importance*

The permutation feature importance is defined to be the decrease in a model score when a single feature value is randomly shuffled. This procedure breaks the relationship between the feature and the target, thus the drop in the model score is indicative of how much the model depends on the feature. This technique benefits from being model agnostic and can be calculated many times with different permutations of the feature.

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture4.png" alt="linearly separable data">

From this approach, we could see that the plot confirms what we have seen above, that the first 3 variables are the most important. The most interesting aspect is being unaccompanied minor(vul_minor) and ethnicity(ethnic) which are added into the most 10 important variables. One more nice feature about rfpimp is that it contains functionalities for dealing with the issue of collinear features.

### Drop Column Feature Importance

This approach requires to drop one feature from the training data and compare this model’s feature importance with all features model. This process will be performed for all features. This technique is assumed to be more accurate feature importance than permutation and

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture5.png" alt="linearly separable data">

Firstly, negative importance, in this case, infers that eliminating given features from the model improves the performance.  The top 10 features stay the same. While it is surprising that by removing expenditure for transport, the performance boost can be observed, though it was among the most important variable in previous approaches. By observing other least important features, it can be extrapolated that it matches quite well with the least important variables from previous approaches.

*Shapley Additive explanations (SHAP)*

The goal of SHAP is to explain the prediction of an instance x by computing the contribution of each feature to the prediction. The SHAP explanation method computes Shapley values from coalitional game theory. The feature values of a data instance act as players in a coalition. Shapley values tell us how to fairly distribute the “payout” (= the prediction) among the features. A player can be an individual feature value, e.g. for tabular data. A player can also be a group of feature values. For example, to explain an image, pixels can be grouped to superpixels and the prediction distributed among them. One innovation that SHAP brings to the table is that the Shapley value explanation is represented as an additive feature attribution method, a linear model.

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture16png.png" alt="linearly separable data">

The global interpretation methods include feature importance, feature dependence, interactions, clustering and summary plots. With SHAP, global interpretations are consistent with the local explanations, since the Shapley values are the “atomic unit” of the global interpretations.

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture7.png" alt="linearly separable data">

The figure above illustrates the SHAP values of a model's output to explain how features impact the output of the model. For fam_origin, mean(SHAP) is about 0.08 on Class 3(Host), and (0.15-0.08)=0.07 on Class 0 (Conflict IDPs), it means fam_origin influence predicting Class 3 and Class 0  quite the same. Fam_origin is also the most influential feature globally. As can be seen from the graph due to the imbalanced structure of the data, Class 1 and Class 2 covers a tiny fraction in the bar charts. Class 1 and Class 2 covers less than 0.5% of the whole dataset.

*CatBoost  model*

CatBoost is based on gradient boosting. It is a machine learning algorithm that allows users to quickly handle categorical features for a large data set and this differentiates it from XGBoost & LightGBM. CatBoost has also a unique advancement in the implementation of ordered boosting. Both techniques help to fight a prediction shift caused by a special kind of target leakage present in all existing implementations of gradient boosting algorithms. CatBoost provides the functionality to calculate the effect of instances from the training dataset on the optimized metric values. In the feature importance function, you can get the SHAP values, just by mentioning type=’ShapValues’

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture8.png" alt="linearly separable data">

The CatBoost SHAP added new variables to the feature importance. Each class is homogeneously distributed within the features, which means that contribution is normalized based on the total datasets. It can be observed top 10 the most important features are quite similar to Random Forest SHAP, while some orders could be different

## Conclusion

By comparing all the techniques of identifying feature importance, it can be concluded that SHAP provides more robust performance compared with other techniques.  SHAP also has a solid theoretical foundation in game theory. The difference of CatBoost from Random Forest SHAP Value, CatBoost uses Stratified cross-validation. It increases the weight of underrepresented classes, to avoid false-positive predictions. However, the given data set is not structured as a typical imbalanced dataset, since there is no high risk of predicting wrong Class 1 and Class 2. Predicting right Class 3 and Class 0 is sufficient. Thus, I assume Random forest SHAP outperforms CatBoost SHAP.

You can find the code used for this article on my [GitHub](https://github.com/assamidanov/Important-Featues-of-IDPs-). Please feel free to contact me if you need any further information.

### References:

1.	https://christophm.github.io/interpretable-ml-book/shap.html
2.	Lundberg, Scott M., and Su-In Lee. “A unified approach to interpreting model predictions.” Advances in Neural Information Processing Systems. 2017
3.	https://shap.readthedocs.io/en/latest/
4.	https://catboost.ai/docs/concepts/tutorials.html\
5. https://www.afghanistan-analysts.org/more-violent-more-widespread-trends-in-afghan-security-in-2017/idps-2017-10-unocha-screen-shot-2017-10-27-at-13-25-39/
