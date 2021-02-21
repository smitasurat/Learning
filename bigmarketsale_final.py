# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:55:32 2021

@author: Smita Shah
"""
# Task 1: Understand the Problem Statement 
# Task 2: Import Libraries and data-sets
# Task 3: Perform Exploratory Data Analysis
# Task 4: Perform Data Visualization
# Task 5: Create Training And Testing Datasets
# Task 6: Train And Evaluate A Linear Regression Model
# Task 7: Train And Evaluate an Artificial Neural Networks Model
# Task 8: Train And Evaluate A Random Forest and Decision Tree Regressors
# Task 9: Understand the difference between regression KPIs
# Task 10: Calculate regression model KPIs    

# Task 2: Import Libraries and data-sets
import os
os.getcwd()
os.chdir("F:\\avjobhackathon_practice")
os.getcwd()

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
# Import library for VIF   checking multicolinearity test
from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np
import seaborn as sns
sns.set( palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf




pd.set_option('max_columns', None)
pd.set_option('display.float_format',lambda x:'%.3f'%x)
pd.set_option('display.max_columns',500)
np.set_printoptions(precision=3)

import warnings
warnings.filterwarnings('error',category=statsmodels.tools.sm_exceptions.HessianInversionWarning)
warnings.filterwarnings('error',category=statsmodels.tools.sm_exceptions.ConvergenceWarning)
warnings.filterwarnings('error',category=RuntimeWarning)
warnings.filterwarnings('ignore',category=UserWarning)

#READ DATA start====================
try:
    train = pd.read_csv('train_v9rqX0R.csv')
    print("train dataset has {} samples with {} features each.".format(*train.shape))
    test = pd.read_csv('test_AbJTz2l.csv')
    print("test dataset has {} samples with {} features each.".format(*test.shape))
    sub = pd.read_csv('sample_submission_8RXa3c6.csv')
    print("submission dataset has {} samples with {} features each.".format(*sub.shape))
  
except:
    print("Dataset could not be loaded. Is the dataset missing?")
#READ DATA :END======================
# 

# Task 3: Perform Exploratory Data Analysis
train.head()
# Task 3a.checking for missing data
train.isnull().sum()
train[['Item_Weight','Outlet_Size']].isnull().sum()
train.describe().transpose()
#checking outliner count
train['Item_Visibility'][train['Item_Visibility']==train['Item_Visibility'].max()]
#only one record no 854 854   0.328
train['Item_Outlet_Sales'][train['Item_Outlet_Sales']==train['Item_Outlet_Sales'].max()]
#only one record no 7188   13086.965
# Imputing with Mean
train['Item_Weight']= train['Item_Weight'].fillna(train['Item_Weight'].mean())
test['Item_Weight']= test['Item_Weight'].fillna(test['Item_Weight'].mean())
#Imputing with Mode
print(train['Outlet_Size'].value_counts())
print('******************************************')
print(test['Outlet_Size'].value_counts())
train['Outlet_Size']= train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
test['Outlet_Size']= test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])

train['Item_Fat_Content'].value_counts()
test['Item_Fat_Content'].value_counts()
# We see there are some irregularities in the column and it is needed to fix them.
train['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
test['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
train['Item_Fat_Content']= train['Item_Fat_Content'].astype(str)
plt.figure(figsize=(8,5))
sns.countplot('Item_Fat_Content',data=train,palette='ocean')
# The Items bought are more of Low Fat.
plt.figure(figsize=(25,7))
sns.countplot('Item_Type',data=train,palette='spring')
# Fruits and Vegetables are largely sold as people tend to use them on daily purpose.
# Snack Foods too have good sales.
plt.figure(figsize=(8,5))
sns.countplot('Outlet_Size',data=train,palette='summer')
# The Outlets are more of Medium Size
sns.countplot('Outlet_Location_Type',data=train,palette='autumn')
# The Outlets are maximum in number in Tier 3 Cities
plt.figure(figsize=(8,5))
sns.countplot('Outlet_Type',data=train,palette='twilight')
# The Outlets are more of Supermarket Type1
plt.figure(figsize=(10,8))
sns.barplot(y='Item_Type',x='Item_Outlet_Sales',data=train,palette='flag')
# The products available were Fruits-Veggies and Snack Foods but the sales of Seafood and Starchy Foods seems higher and hence the sales can be improved with having stock of products that are most bought by customers.
# Step 5: Building Model
train.head()
df_train=train.copy()
df_test=test.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_mod = df_train.select_dtypes(include='object').columns
for i in var_mod:
    df_train[i] = le.fit_transform(df_train[i])
    
for i in var_mod:
    df_test[i] = le.fit_transform(df_test[i])
    
train.columns


# checking for duplicate records
train[train.duplicated(subset=None,keep='first')].count()
test[test.duplicated(subset=None,keep='first')].count()
# NO duplicates row found
# The regression has five key assumptions:
# Linear relationship.
# Multivariate normality......vif
# No or little multicollinearity.
# No auto-correlation.....
# Homoscedasticity.

# Seperate Features and Target
X= df_train.drop(columns = ['Item_Outlet_Sales'], axis=1)
y= df_train['Item_Outlet_Sales']
x_col=X.columns.to_list()


# 1. Linearity
# Linear regression needs the relationship between the independent and dependent variables to be linear. Let's use a pair plot to check the relation of independent variables with the Sales variable
# visualize the relationship between the features and the response using scatterplots
p = sns.pairplot(df_train, x_vars=x_col, y_vars='Item_Outlet_Sales', size=7, aspect=0.7)
# check for linear  relationship in graph
#preparation to test other realationship

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_SC= sc.fit_transform(X)
# y=np.array(y).reshape(-1,1)
# y_sc= sc.fit_transform(y)
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,test_size=0.25)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_train)
print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))

# 2. Mean of Residuals

# Residuals as we know are the differences between
#  the true value and the predicted value. 
#  One of the assumptions of linear regression is
#  that the mean of the residuals should be zero. So let's find out.


residuals = y_train.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))
# Very close to zero so all good here.


# 3. Check for Homoscedasticity
# Homoscedasticity means that the residuals have 
# equal or almost equal variance across the regression line. 
# By plotting the error terms with predicted terms we can check that there should not be any pattern in the error terms.

# Detecting heteroscedasticity!
# 1.Graphical Method: Firstly do the regression analysis and then plot the error terms against the predicted values( Yi^). If there is a definite pattern (like linear or quadratic or funnel shaped) obtained from the scatter plot then heteroscedasticity is present.

# Now we will be applying tests.
# A tip is to keep in mind that if we want 95% confidence on our findings and tests then the p-value should be less than 0.05 to be able to reject the null hypothesis. Remember, a researcher or data scientist would always aim to reject the null hypothesis.


# Goldfeld Quandt Test
# Checking heteroscedasticity : Using Goldfeld Quandt
#  we test for heteroscedasticity.

# Null Hypothesis: Error terms are homoscedastic
# Alternative Hypothesis: Error terms are heteroscedastic.

import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
goldtest = sms.het_goldfeldquandt(residuals, X_train)
lzip(name, goldtest)

# Out[291]: 
# [('F statistic', 0.9592570186404753), ('p-value', 0.8797271693253719)]
# Since p value is more than 0.05 in Goldfeld Quandt Test, we can't reject it's null hypothesis that error terms are homoscedastic. Good.

# Bartlett’s test tests the null hypothesis 
# that all input samples are from populations with equal variances.
# from scipy.stats import bartlett
# # following is giving error so use y_train instaed of x train need to check

# # test = bartlett( X_train,residuals)


# barlett_test = bartlett( y_train,residuals)
# print(barlett_test)
# Since p value is quite less than 0.05 in Bartlett, it's null hypothesis that error terms are homoscedastic gets rejected, that's not good for a regression.


# 4. Check for Normality of error terms/residuals

p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')



# The residual terms are pretty much normally distributed 
# for the number of test points we took.
#  Remember the central limit theorem which says that
#  as the sample size increases the distribution tends to be normal. A skew is also visible from the plot. It's very difficult to get perfect curves, distributions in real life data.

# 5. No autocorrelation of residuals
# When the residuals are autocorrelated, it means that
#  the current value is dependent of the previous 
#  (historic) values and that there is 
#  a definite unexplained pattern in the Y variable that shows up in the error terms. Though it is more evident in time series data.

# In plain terms autocorrelation takes place when 
# there's a pattern in the rows of the data. 
# This is usual in time series data as there is a pattern
#  of time for eg. Week of the day effect which is a very famous pattern seen in stock markets where people tend to buy stocks more towards the beginning of weekends and tend to sell more on Mondays. There's been great study about this phenomenon and it is still a matter of research as to what actual factors cause this trend.
# There should not be autocorrelation in the data so the error terms should not form any pattern.
 
plt.figure(figsize=(10,5))
p = sns.lineplot(y_pred,residuals,marker='o',color='blue')
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-1500,5000)
plt.xlim(1500,4000)
p = sns.lineplot([0,26],[0,0],color='red')
p = plt.title('Residuals vs fitted values plot for autocorrelation check')

# Checking for autocorrelation To ensure the absence of autocorrelation we use Ljungbox test.
# Null Hypothesis: Autocorrelation is absent.
# Alternative Hypothesis: Autocorrelation is present.
from statsmodels.stats import diagnostic as diag
min(diag.acorr_ljungbox(residuals , lags = 40)[1])

# Out[303]: 0.2584645257937863

# Since p value is more  than 0.05 we accept the null hypothesis
#  that error terms are not autocorrelated.

import statsmodels.api as sm
# autocorrelation
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.show()

# partial autocorrelation
sm.graphics.tsa.plot_pacf(residuals, lags=40)
plt.show()

# The results show no  signs of autocorelation 
# since there are no spikes outside the red confidence interval region. This could be a factor of seasonality in the data.

# 6. No perfect multicollinearity
# In regression, multicollinearity refers to the extent
 # to which independent variables are correlated.
 # Multicollinearity affects the coefficients and p-values,
 # but it does not influence the predictions, 
 # precision of the predictions, and the goodness-of-fit 
 # statistics. 
 # If your primary goal is to make predictions, and you don’t need to understand the role of each independent variable, you don’t need to reduce severe multicollinearity.¶

plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(df_train.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

# Import library for VIF   checking multicolinearity test
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(X)

# vif more than five show multicolinearity
# Out[30]: 
#                     variables    VIF
# 0             Item_Identifier  4.081
# 1                 Item_Weight 10.292
# 2            Item_Fat_Content  1.601
# 3             Item_Visibility  2.777
# 4                   Item_Type  4.041
# 5                    Item_MRP  6.139
# 6           Outlet_Identifier 15.515
# 7   Outlet_Establishment_Year 56.915
# 8                 Outlet_Size  7.958
# 9        Outlet_Location_Type 16.598
# 10                Outlet_Type  8.435

train.columns
# removing multi colinearirty
train.columns
x1=df_train[['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type']].copy()
f1_v1=calc_vif(x1)

# train=train.drop(columns = ['Outlet_Establishment_Year'], axis=1)
# test=test.drop(columns = ['Outlet_Establishment_Year'], axis=1)
x1=df_train[['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP',
        'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type']].copy()
f1_v1=calc_vif(x1)

import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns F and P-value
#fvalue, pvalue = stats.f_oneway(d['In-aisle'])

import statsmodels.api as sm
from statsmodels.formula.api import ols

regr = LinearRegression()
regr.fit(X_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

X = sm.add_constant(X_train) # adding a constant
 
model = sm.OLS(y_train, X).fit()
print(model.summary())
# Durbin-Watson:                   1.991... implies that residuals are not correlatd
# Intercept: and  Coefficients:  generated by sklearn and ols is same



# feature creation
train['Outlet_Years'] = 2021 - train['Outlet_Establishment_Year']
test['Outlet_Years'] = 2021 - test['Outlet_Establishment_Year']

# Create a broad category of Type of Item
#Years:

train['Item_Type_Combined'] = train['Item_Identifier'].apply(lambda x: x[0:2])
train['Item_Type_Combined'] = train['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})

test['Item_Type_Combined'] = test['Item_Identifier'].apply(lambda x: x[0:2])
test['Item_Type_Combined'] = test['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})

# Modify categories of Item_Fat_Content
#Mark non-consumables as separate category in low_fat:
train.loc[train['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
train['Item_Fat_Content'].value_counts()

#Mark non-consumables as separate category in low_fat:
test.loc[test['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
test['Item_Fat_Content'].value_counts()

# Numerical and One-Hot Coding of Categorical variables

#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
train['Outlet'] = le.fit_transform(train['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
    
#One Hot Coding:
train = pd.get_dummies(train, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'],drop_first=True)

#New variable for outlet
test['Outlet'] = le.fit_transform(test['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    test[i] = le.fit_transform(test[i])
    
#One Hot Coding:
test = pd.get_dummies(test, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'],drop_first=True)

# Step 6: Building Model
train.columns
train.info()
train = train.select_dtypes(exclude='object')
test = test.select_dtypes(exclude='object')

# Seperate Features and Target
# Seperate Features and Target
X= train.drop(columns = ['Item_Outlet_Sales'], axis=1)
y= train['Item_Outlet_Sales']


calc_vif(X)

# 20% data as validation set
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=22)

# Model Building
features= X.columns
LR = LinearRegression(normalize=True)
LR.fit(X_train,y_train)
y_pred = LR.predict(X_valid)
coef = pd.Series(LR.coef_,features).sort_values()

# Barplot for coefficients
plt.figure(figsize=(8,5))
sns.barplot(LR.coef_,features)

# Item Visibility, Outlet_Type, Outlet_Size, Outlet_Location_Type are
# the most important attributes to determine Item_Outlet_Sales.
 
MSE= metrics.mean_squared_error(y_valid,y_pred)
from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)

# ANOVA

algos = [LinearRegression(),  Ridge(), Lasso(),
          KNeighborsRegressor(), DecisionTreeRegressor(),RandomForestRegressor()]

names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
         'K Neighbors Regressor', 'Decision Tree Regressor','RandomForestRegressor']

rmse_list = []

for name in algos:
    model = name
    model.fit(X_train,y_train)
    y_pred = model.predict(X_valid)
    MSE= metrics.mean_squared_error(y_valid,y_pred)
    rmse = np.sqrt(MSE)
    rmse_list.append(rmse)
    
evaluation = pd.DataFrame({'Model': names,'RMSE': rmse_list})

evaluation
# with sklearn

submission = pd.read_csv('sample_submission_8RXa3c6.csv')
model = RandomForestRegressor()
model.fit(X_train,y_train)

final_predictions = model.predict(test)
submission['Item_Outlet_Sales'] = final_predictions

#only positive predictions for the target variable
submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x<0 else x)
submission.to_csv('big_rf_submission_vif_f2.csv', index=False)

rf = RandomForestRegressor()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# create the random grid
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid,
                               scoring='neg_mean_squared_error',
                               n_iter = 10, cv = 5, 
                               verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)

X_train.info()
best_parameter=rf_random.best_params_
rf_random.best_estimator_
rf_random.best_score_
predictions=rf_random.predict(X_valid)

sns.displot(y_valid-predictions)
plt.scatter(y_valid,predictions)

print('MSE:', metrics.mean_squared_error(y_valid, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))

submission = pd.read_csv('sample_submission_8RXa3c6.csv')
final_predictions = rf_random.predict(test)
submission['Item_Outlet_Sales'] = final_predictions
submission[submission['Item_Outlet_Sales'] < 0]

submission.to_csv('big_rfcv_vif_nf2_submission.csv', index=False)

# rf1=RandomForestRegressor(n_estimators= 300,
# min_samples_split= 100,
# min_samples_leaf= 5,
# max_features= 'auto',
# max_depth= 15)

rf1=rf_random.best_estimator_
rf1.fit(X_train,y_train)
rf1_predictions=rf1.predict(X_valid)

print('MSE:', metrics.mean_squared_error(y_valid, rf1_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, rf1_predictions)))

feats_imp = pd.DataFrame(rf1.feature_importances_, index=X_train.columns, columns=['FeatureImportance'])
feats_imp = feats_imp.sort_values('FeatureImportance', ascending=True)

feats_imp.plot(kind='barh', figsize=(12,6), legend=False)
plt.title('Feature Importance from RandomForest Regressor')
# sns.despine(left=True, bottom=True)
# plt.gca().invert_yaxis()

submission = pd.read_csv('sample_submission_8RXa3c6.csv')
final_predictions = rf1.predict(sub_12)
submission['Item_Outlet_Sales'] = final_predictions
submission[submission['Item_Outlet_Sales'] < 0]

submission.to_csv('big_rf1_vif_f1_submission.csv', index=False)
sub_12.info()
# ann from rgression guided project
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation ,Dropout
from tensorflow.keras.optimizers import Adam


ANN_model=keras.Sequential()
ANN_model.add(Dense(50,input_dim=10))
ANN_model.add(Activation('relu'))

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(50))
ANN_model.add(Activation('linear'))
ANN_model.add(Dense(1))

ANN_model.compile(loss='mse',optimizer='adam')
ANN_model.summary()


ANN_model.compile(optimizer='Adam',loss='mean_squared_error')
epoch_hist=ANN_model.fit(X_train,y_train,epochs=91,batch_size=20)

result=ANN_model.evaluate(X_valid,y_valid)
accuracy_ann=1-result
print('accuracy : {}'.format(accuracy_ann))

epoch_hist.history.keys()
# print(dir(plt))
epoch_hist.history.head()
plt.plot(epoch_hist.history['loss'],marker='')
plt.title("Model loss Prograss during training")
plt.xlabel('epoch')
plt.ylabel("Training loss ")
plt.legend(["Training loss "])
plt.show()
ann_prediction=ANN_model.predict(test)
submission = pd.read_csv('sample_submission_8RXa3c6.csv')

submission['Item_Outlet_Sales'] = ann_prediction
submission[submission['Item_Outlet_Sales'] < 0]

submission.to_csv('big_ann_vif_f1_submission.csv', index=False)


import lightgbm as lgb
params={}

params['boosting_type'] = 'gbdt'
params['max_depth'] = 15
params['learning_rate'] = 0.01
params['n_estimators'] = 300


params['objective'] = 'regression'
params['subsample'] = 0.7
params['metric']='mse'

params['random_state'] = 42
params['colsample_bytree']=0.7
params['min_data_in_leaf'] = 5
params['reg_alpha'] = 1.7
params['reg_lambda'] = 1.11
#params['scale_pos_weight']=4.5 
#Cannot set is_unbalance and scale_pos_weight at the same time
# params['is_unbalance']=True

lgbm = lgb.LGBMRegressor(**params)
#
#lgbm.fit(x_train, y_train,categorical_feature=catcol)
lgbm.fit(X_train, y_train)

lgbm_predict=lgbm.predict(X_valid)
mse_lgbm=mean_squared_error(y_valid,lgbm_predict)

print('MSE:', metrics.mean_squared_error(y_valid, lgbm_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgbm_predict)))


lgbm_random = RandomizedSearchCV(estimator = lgbm, 
                               param_distributions = random_grid,
                               scoring='neg_mean_squared_error',
                               n_iter = 10, cv = 10, 
                               verbose=2, random_state=42, n_jobs = 1)
lgbm_random.fit(X_train,y_train)

lgbm_cv=lgbm_random.best_estimator_

predictions=lgbm_random.predict(X_valid)

lbm_parameter=lgbm_random.best_params_
rf_random.best_estimator_
rf_random.best_score_


sns.displot(y_valid-predictions)
plt.scatter(y_valid,predictions)

print('MSE:', metrics.mean_squared_error(y_valid, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))

submission = pd.read_csv('sample_submission_8RXa3c6.csv')
final_predictions = lgbm_random.predict(sub_12)
submission['Item_Outlet_Sales'] = final_predictions
submission[submission['Item_Outlet_Sales'] < 0]

submission.to_csv('big_lgbmcv_vif_f1_submission.csv', index=False)

