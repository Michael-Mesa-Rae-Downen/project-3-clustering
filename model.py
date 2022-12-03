# importing of all needed libraries and modules.  
from math import sqrt
from pathlib import Path
from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PolynomialFeatures
import env
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wrangle as w

# pull the data from the wrangle file
df=w.wrangle_zillow()

# split the data
train, validate, test = w.split_data(df)

# scale the data
train_scaled, validate_scaled, test_scaled=w.scale_data(train, validate, test)

# assign the train variables
X_train_scaled = train_scaled[['bedrooms','bathrooms', 'sq_feet']]
y_train=train[['logerror']]

#assign the validate variables
X_validate_scaled=validate_scaled[['bedrooms','bathrooms', 'sq_feet']]
y_validate=validate[['logerror']]

# assign the test variables
X_test_scaled=test_scaled[['bedrooms','bathrooms', 'sq_feet']]
y_test=test[['logerror']]

# BASELINE
#Use RMSE on both the mean and median.

# Create a function to get baseline.
def baseline(X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test):
    # We need y_train and y_validate (and test) to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    # 1. Predict home_value_pred_mean
    logerror_pred_mean = y_train['logerror'].mean()
    y_train['logerror_pred_mean'] = logerror_pred_mean
    y_validate['logerror_pred_mean'] = logerror_pred_mean

    # 2. compute home_value_pred_median
    logerror_pred_median = y_train['logerror'].median()
    y_train['logerror_pred_median'] = logerror_pred_median
    y_validate['logerror_pred_median'] = logerror_pred_median

    # 3. RMSE of home_value_pred_mean
    rmse_train = mean_squared_error(y_train[['logerror']], y_train.logerror_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate[['logerror']], y_validate.logerror_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", (rmse_train), 
      "\nValidate/Out-of-Sample: ", (rmse_validate))

    # 4. RMSE of home_value_pred_median
    rmse_train = mean_squared_error(y_train[['logerror']], y_train.logerror_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate[['logerror']], y_validate.logerror_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", (rmse_train), 
      "\nValidate/Out-of-Sample: ", (rmse_validate))

# create the OLS model object
def ols():
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_scaled, y_train.logerror)

    # predict train
    y_train['logerror_pred_lm'] = lm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm)**(1/2)

    # predict validate
    y_validate['logerror_pred_lm'] = lm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm)**(1/2)

    print("RMSE for OLS\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)

# create the ols viz    
def ols_viz():
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.logerror, color='green', alpha=.5, label="Absolute Value Log Error")
    plt.hist(y_validate.logerror_pred_lm, color='blue', alpha=.5, label="Validate")
    plt.xlabel("Log Error")
    plt.ylabel("Homes")
    plt.title("Comparing the Distribution of Absolute Value Log Error OLS Model")
    plt.legend()
    plt.show()    

# create the polynomial regression model object

def poly():
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 = pf.transform(X_test_scaled)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.logerror)

    # predict train
    y_train['logerror_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm2)**(1/2)

    # predict validate
    y_validate['logerror_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm2)**(1/2)

    print("RMSE for Poly 2\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

# create the poly viz
def poly_viz():
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.logerror, color='green', alpha=.5, label="Absolute Value Log Error")
    plt.hist(y_validate.logerror_pred_lm2, color='blue', alpha=.5, label="Validate")
    plt.xlabel("Log Error")
    plt.ylabel("Homes")
    plt.title("Comparing the Distribution of Absolute Value Log Error Polynomial Model")
    plt.legend()
    plt.show()
    
#Create a function for the LassoLars regression model object

def lasso_lars():
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train_scaled, y_train.logerror)

    # predict train
    y_train['logerror_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lars)**(1/2)

    # predict validate
    y_validate['logerror_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lars)**(1/2)

    # predict test
    #y_test['logerror_pred_lars'] = lars.predict(X_test_scaled)

    # evaluate: rmse
    #rmse_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
      #"\nTesting/Out-of-Sample Performance: ", rmse_test)

# create the lars viz        
def lars_viz():
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.logerror, color='green', alpha=.5, label="Absolute Value Log Error")
    plt.hist(y_validate.logerror_pred_lars, color='blue', alpha=.5, label="Validate")
    plt.xlabel("Log Error")
    plt.ylabel("Homes")
    plt.title("Comparing the Distribution of Absolute Value Log Error on Lasso Lars Model")
    plt.legend()
    plt.show()   
    
 #Create a function for the Tweedie regression model object
def tweedie():
    glm = TweedieRegressor(power=0, alpha=1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train_scaled, y_train.logerror)

    # predict train
    y_train['logerror_pred_glm'] = glm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_glm)**(1/2)

    # predict validate
    y_validate['logerror_pred_glm'] = glm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_glm)**(1/2)

    # predict test
    #y_test['logerror_pred_glm'] = glm.predict(X_test_scaled)
    
    # evaluate: rmse
    #rmse_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_glm)**(1/2)
    
    print("RMSE for Tweedie\nTraining/In-Sample: ", rmse_train,
          "\nValidation/Out-of-Sample: ", rmse_validate)
          #"\nTesting/Out-of-Sample Performance: ", rmse_test)   
    
# create the tweedie viz
def tweedie_viz():
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.logerror, color='green', alpha=.5, label="Absolute Value Log Error")
    plt.hist(y_validate.logerror_pred_glm, color='blue', alpha=.5, label="Validate")
    plt.xlabel("Log Error")
    plt.ylabel("Homes")
    plt.title("Comparing the Distribution of Absolute Value Log Error on Tweedie Model")
    plt.legend()
    plt.show()    

def tweedie_viz2():
    plt.figure(figsize=(16,8))
    #plt.hist(y_validate.logerror, color='green', alpha=.5, label="Absolute Value Log Error")
    plt.hist(y_validate.logerror_pred_glm, color='blue', alpha=.5, label="Validate")
    plt.xlabel("Log Error")
    plt.ylabel("Homes")
    plt.title("Comparing the Distribution of Absolute Value Log Error on Tweedie Model")
    plt.legend()
    plt.show()

def r2_score():
    evs = explained_variance_score(y_train.logerror, y_train.logerror_pred_lm)
    print('The R2 or Explained Variance = ', round(evs,3))
    
    
 # create the ols test model object
def ols_test():
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_scaled, y_train.logerror)

    # predict test
    y_test['logerror_pred_lm'] = lm.predict(X_test_scaled)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_lm)**(1/2)

    print("RMSE for OLS\nTesting/Out-of-Sample Performance: ", rmse_test)
   
 # create best model viz    
def ols_test_viz():
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.logerror, color='green', alpha=.5, label="Absolute Value Log Error")
    plt.hist(y_test.logerror_pred_lm, color='yellow', alpha=.5, label="Test")
    plt.xlabel("Log Error")
    plt.ylabel("Homes")
    plt.title("Comparing the Distribution of Absolute Value Log Error to OLS Test Data")
    plt.legend()
    plt.show()    
    
# create viz of actual distribution
def actual_viz():
    plt.figure(figsize=(16,8))
    plt.hist(y_train.logerror, color='green', alpha=.5, label="Train")
    plt.hist(y_validate.logerror, color='yellow', alpha=.5, label="Validate")
    plt.hist(y_test.logerror, color='red', alpha=.5, label="Test")
    plt.xlabel("Log Error")
    plt.ylabel("Homes")
    plt.title("Comparing the Actual Distribution of Absolute Value Log Error on Train, Validate, Test")
    plt.legend()
    plt.show()    
    
  # creat viz of predicted log errorDistribution
def predict_viz():
    plt.figure(figsize=(16,8))
    plt.hist(y_train.logerror_pred_lm, color='green', alpha=.5, label="Train")
    plt.hist(y_validate.logerror_pred_lm, color='yellow', alpha=.5, label="Validate")
    plt.hist(y_test.logerror_pred_lm, color='red', alpha=.5, label="Test")
    plt.xlabel("Log Error")
    plt.ylabel("Homes")
    plt.title("Comparing the Predicted Distribution of Absolute Value Log Error on Train, Validate, Test")
    plt.legend()
    plt.show()  
    
 # compare the models    
def compare_viz():
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.logerror_pred_lm, color='blue', alpha=.5, label="OLS")
    plt.hist(y_validate.logerror_pred_lm2, color='green', alpha=.5, label="Polynomial")
    plt.hist(y_validate.logerror_pred_lars, color='yellow', alpha=.5, label="Lasso Lars")
    plt.hist(y_validate.logerror_pred_glm, color='pink', alpha=.5, label="Tweedie")
    plt.xlabel("Models")
    plt.ylabel("Homes")
    plt.title("Comparing the Models of Absolute Value Log Error")
    plt.legend()
    plt.show()