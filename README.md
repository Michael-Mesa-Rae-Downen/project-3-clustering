# Clustering and Logerror
#  
# Project Description
-Zillow wants insight on what is driving the errors in the Zestimates

# Project Goal
- Explore the effects of calculatedfinishedsquarefeet and bath_bed_ratio on the dependent variable, logerror, on Single Family Properties that had a transaction in 2017.
- Use clustering to explore the interactions.  Construct a ML regression model that can help find what is driving errors in the Zestimates.

# Initial Thoughts
My initial hypothesis is that drivers of logerror will likely be main features of most homes, such as the number of bedrooms, bathrooms, and square footage, age, and possibly tax values.

# Plan

- Acquire data from zillow database in SQL

- Prepare data by dropping unnecessary columns, removing nulls, renaming columns, removing outliers, encoding, and optimizing data types.

- Split and scale the data.

- Use statistical tests and visualizations when exploring the data to find drivers or factors that might influence logerror.
 
- Explore data in search of drivers of logerror using cluster on a combination of features and answer the following:

> Is the number of bathrooms related to logerror?

> Is the number of bedrooms related to logerror?

> Is the number of square feet related to logerror?

> Will the clusters help in determining drivers related to log error?

- Based on the exploration and clustering, develop a regression model that will help identify drivers of logerror.

> Use drivers identified in explore to build predictive models

> Evaluate models on train and validate data

> Select best model based on highest accuracy

> Evaluation of best model on test data

- Draw conclusions

# Data Dictionary

| Feature | Definition |
| :- | :- |
| sq_feet	| Calculated total finished living area of the home |
| bedrooms |  Number of bedrooms in home |
| bedrooms |  Number of bedrooms in home |
| logerror | Zillow zestimate log error of sale price |


# Steps to Reproduce
1. Clone this repo
2. Acquire the data from SQL database
3. Place data in file containing the cloned repo
4. Run notebook

# Takewaways and Conclusions
Exploration Summary¶
    'bedrooms' was found to be a driver of 'logerror'
    'bathrooms' was found to be a driver of 'logerror'
    'sq_feet' was found to be a driver of 'logerror'
    Observations in cluster 3 was found to be a driver of 'logerror'

Modeling Summary
    Using the OLS, Polynomial (degree 2), Lasso Lars, and Tweedie regression models help to determine the best model to use.
    Although the variance was small, the best model in this case was to use the OLS model.
    Comparing them visually did not help in determining the best model.
    Comparing them against the RMSE was in our best interest.
 
# Recommendations and Next Steps
Recommendations¶
      We should reconsider how we are approaching this problem.
      Further explore our data for feature selection that will yield more substantial results.

Next Steps
    Explore our data with a new approach.
    Possibly focus more on the location of properties while keeping property size in consideration.

