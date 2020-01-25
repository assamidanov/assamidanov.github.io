---
title: "Identifying Important Features Using Household Survey Data"
date: 2019-01-04
tags: [machine learning, data science, feature engineering, feature importance, random forest, catboost]
#header:
  # image: "/images/project/bla/bla/" map of Afganistan
excerpt: "Machine Learning, Feature Importance, Data Science"
---

## Background Information

Afghanistan faces one of the world’s most acute internal displacement crises; resulting of several factors such as protracted conflict, ongoing insecurity, and natural hazards. Displacement has become a familiar survival strategy for many Afghans and, in some cases, an inevitable part of life for two generations. As of 31 December 2018, Afghanistan has 2,598,000 total number of internally displaced persons (IDPs).

Displacement affects all individuals differently with needs, vulnerabilities and protection risks evolving due to exhaustion of coping mechanisms and only basic emergency assistance provided following initial displacement. Inadequate shelter, food insecurity, insufficient access to sanitation and health facilities, as well as a lack of protection, often result in precarious living conditions that jeopardize the well-being and dignity of affected families.

## Aim of the Project

The survey was taken by Arqaam to monitor and evaluate the NGOs’ humanitarian assistance projects. It is also directed towards understanding the status-quo before the beginning of the project, and the impact the project had over time.
The survey has 237 comprehensive questions and 11260 respondents. The survey was taken in a randomly stratified approach. Jalalabad and Herat were chosen as a target region. It can be claimed with its variety of ethnic groups and displacement specifications both cities can be representative of the whole of Afghanistan.

This project aims to perform a machine-learning algorithm to predict the displacement status of the respondents based on the responses from the questionnaire. Then, to determine important features that define displacement status of residents by using state of art techniques.  I used two Machine Learning algorithms like Random Forest Classifier and CatBoost Classifier to identify these important features. I also used 4 feature importance techniques: Default feature importance, Permutation feature importance, Drop Column feature importance and Shapley Additive explanations (SHAP).By comparing the outcomes of these techniques, I will determine the most important and the least important features.

## Explanatory Data Analysis

<img src="{{ site.url }}{{ site.baseurl }}/images/Arqaam/Picture1.png" alt="linearly separable data">
