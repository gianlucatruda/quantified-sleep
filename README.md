# Quantified Sleep

An exploration of techniques for observational n-of-1 studies.

## Overview

![overview diagram](img/QuantifiedSleepOverview.jpg)

This paper combines techniques from across disciplines and applies them in an observational n-of-1 Quantified-Self (QS) study to build a descriptive model of sleep quality. A total of 472 days of the author's sleep data, collected nightly with an Oura ring, were combined with a variety of lifestyle, environmental, and psychological data harvested from multiple sensors and manual logs.

Observational n-of-1 QS projects pose a number of specific challenges: heterogeneous data sources with many missing values; few observations and many features, resulting in overparameterised models; and systems composed of dynamic feedback loops that exacerbate human biases. This paper directly addresses these problems through two main contributions: an end-to-end QS pipeline for observational studies, and a wide-ranging exploration of complementary techniques for overcoming each of the challenges of n-of-1 QS projects. 

Sleep quality is one of the most challenging modelling targets in QS research, due to high noise and a high number of weakly-contributing factors, meaning that approaches from this paper will generalise to most other n-of-1 QS projects. 

Techniques are presented for combining heterogeneous data sources and engineering day-level features from different data types and frequencies, including manually-tracked event logs and automatically-sampled weather and geo-spatial data. Relevant statistical analyses for outliers, normality, (auto)correlations, stationarity, and missing data are detailed, along with a proposed method for hierarchical clustering to identify correlated groups of features.
The missing data was overcome using a combination of knowledge-based and statistical techniques, including several multivariate imputation algorithms. 

"Markov unfolding" is presented as a technique for collapsing the time series into a collection of independent observations for modelling, thus incorporating historical data. 
The final model was interpreted in two key ways: by inspecting the internal beta-parameters, and using the SHAP framework, which works on any "black box" model. These two interpretation techniques were combined to produce a list of the 16 most-predictive features.

By combining contemporary techniques, this project identified the factors that most-impact my sleep, demonstrating that an _observational_ study can greatly narrow down the number of features that need to be considered in interventional n-of-1 QS research.


