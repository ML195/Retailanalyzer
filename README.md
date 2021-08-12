# Retailanalyzer
This Python project implements a solution for the following three issues using the Online Retail II UCI data set (https://www.kaggle.com/mashlyn/online-retail-ii-uci):
1. What are the expected weekly aggregated sales for the next four weeks?
1. Which customers will buy something from the store wihtin the next quarter and which won't?
1. What products can be recommended based on a given customer or based on another product?


## 1. Approach Summary

The figure above shows the general approach used to solve the three issues. First the input data is prepared for further steps via the DataHandler. Depending on the problem, a specific preprocessing strategy can be passed to the DataHandler, including the SalesForecasterPreprocessor, the PurchasePredictorPreprocessor and the ProductRecommenderPreprocessor. This results in a total of three prepared data sets, each involving different data cleaning steps and feature sets specifically built for the following task. A more detailed description of the features is included in the (preliminiary_analysis.pdf). Based on these prepared data sets the tasks are then solved using appropriate models.

### 1. 1. Sales Forecasting
Sales forecasting can be done with an ARIMAForecaster and an LSTMForecaster. For both forecasters GridSearchCV with TimeSeriesSplitCV (6 splits, each testing on 8 future periods) is performed over a predefined search space which can be provided through a hyperparamter grid, such as the ones shown below.

```python
# Hyperparameter Grid for ARIMAForecaster
ARIMA_SETTINGS = {
            'p': [0, 1, 2],
            'd': [1],
            'q': [0, 1, 2],
            'P': [0, 1, 2],
            'D': [0],
            'Q': [0],
            'm': [0, 52]
        }

# Hyperparameter Grid for LSTMForecaster
LSTM_SETTINGS = {
            'neurons_layer_1': [1, 2, 4],
            'neurons_layer_2' : [1, 2, 4],
            'recurrent_dropout': [0, 0.1],
            'epochs': [50, 100, 150, 200],
            'learning_rate': [0.001],
            'batch_size': [1]
    }
```
The hyperparameter configuration leading to the best AIC score (for ARIMA) or the best RMSE (for LSTM) is then used for the final forecasting model fitted on the whole time series.

### 1. 2. Purchase Prediction
Predicting if a customer will buy next quarter is based on the same workflow, where the best model hyperparameters are found using GridSearchCV with StratifiedKFoldCV (5 splits). The metric used for determining the best model is the maximal achieved generalized F-score (fBeta, with a user defined beta) for each prediction threshold (in 0.01 steps). A PurchasePredictor can take every classifier provided by the scikit-learn library that features a `predict_proba()` function. An example setting for searching the best model among two classifiers would look like:

```python
# Hyperparameter Grid for PurchasePredictor
PURCHASE_PREDICTOR_SETTINGS = [
        {
            'CLF_NAME': 'LogisticRegression',
            'HYPERPARAMS': [
                {
                    'penalty': ['l2'],
                    'C': [1, 0.1, 0.01]
                }
            ]
        },
        {
            'CLF_NAME': 'RandomForestClassifier',
            'HYPERPARAMS': [
                {
                    'n_estimators': [100, 200],
                    'min_samples_split': [2, 16, 128],
                }
            ]
        }
    ]
```

### 1. 3. Product Recommendation
For generating product recommendations a kNN-inspired IBCF (item-based collaborative filtering) recommender is implemented. The recommender is evaluated using a random train/test split (5 splits) where each split's test set includes 10 purchased items for each customer, that was removed from the split's training data. Based on that the average hit-rate as well as the average reciprocal hit-rate is calcualted. 

### 1. 4. Additional Notes
* All models and hyperparameter configurations tried in this project are shown in the model_creator.py module (link).
* To see the best models and their achieved scores, have a look at (link). 
* More detailed information about the implemented modules can be found in the code documentation (link). 
* To get a quick overview, see the src/modelcreator/example_usage.py module, which contains an implementaion and explanation of simple usecases showcasing how the modules can be used. 

## 2. Folder Structure
<pre>
<b>.</b>
 <b>|__ data</b>
     <b>|__ processed</b> (stored preprocessed data sets)
     <b>|__ raw</b> (original data sets)
 <b>|__ docs</b> (sphinx project documentation folder)
 <b>|__ models</b></b> (stored and serialized models)
 <b>|__ notebooks</b> (jupyter notebooks used for analysis)
 <b>|__ reports</b>
     <b>|__ evaluations</b> (evaluations and plots for created models)
     <b>|__ results</b> (plots for forecast)
 <b>|__ src</b>
     <b>|__ logs</b> (logging file)
     <b>|__ modelcreator</b> (root package, that contains subpackages and corresponding modules)
     <b>|__ tests</b> (pytest testfiles)
</pre>

## 3. Used Libraries
**Python** - v3.8.8  
**numpy** - v1.19.5  
**pandas** - v1.2.3  
**tensorflow** - v2.5.0  
**scikit-learn** - v0.24.2  
**pmdarima** - v1.8.2  
**statsmodels** - v0.12.2  
**matplotlib** - v3.4.0  
**seaborn** - v0.11.1  
**pytest** - v6.2.4  
**Sphinx** - v4.1.2
