# Clustering and Logerror
#  
# Project Description
-Zillow wants insight on what is driving the errors in the Zestimates

# Project Goal
- Explore the effects of calculatedfinishedsquarefeet and bath_bed_ratio on the dependent variable, logerror, on Single Family Properties that had a transaction in 2017.
- Use clustering to explore the interactions.  Construct a ML regression model that can help find what is driving errors in the Zestimates.

# Initial Thoughts
My initial hypothesis is that drivers of logerror will likely be main features of most homes, such as the number of bedrooms, bathrooms, and square footage, age, and possibly tax values.
Null hypothesis- The independent variables that are explored, modeled, and tested will not be drivers of logerror.
Alternative hypothesis- The independent variables that are explored, modeled, and tested will be drivers of logerror.

# Plan

- Acquire data from zillow database in SQL

- Prepare data by dropping unnecessary columns, removing nulls, renaming columns, removing outliers, encoding, and optimizing data types.

- Split and scale the data.

- Use statistical tests and visualizations when explore the data to find drivers or factors that might influence logerror.
 
- Explore data in search of drivers of logerror using cluster on a combination of features and answer the following:

> Are the number of bedrooms related to logerror?

> Are the number of bathrooms related to logerror?

> Is square footage related to logerror?

> Is tax value of actual structure or lot related to logerror?

- Based on the exploration and clustering, develop a regression model that will help identify drivers of logerror.

> Use drivers identified in explore to build predictive models

> Evaluate models on train and validate data

> Select best model based on highest accuracy

> Evaluation of best model on test data

- Draw conlcusions

# Data Dictionary

| Feature | Definition |
| :- | :- |
| calculatedfinishedsquarefeet	| Calculated total finished living area of the home |
| bath_bed_ratio | Ratio of bathrooms to bedrooms |
| logerror | Zillow zestimate log error of sale price |


# Steps to Reproduce
1. Clone this repo
2. Acquire the data from SQL database
3. Place data in file containing the cloned repo
4. Run notebook

# Takewaways and Conclusions

- There is a positive correlation between the number of bedrooms and tax value of a property.


- There is a positive correlation between the number of bathrooms and tax value of a property.


- There is a positive correlation between the amount of square footage and tax value of a property.


- There is an uneven distribution of properties in the Los Angeles County, Orange County, an Ventura County.

# Recommendations

- Continue to focus on the number of bedrooms, bathrooms, and square footage of homes as drivers of tax value.


- Evaluate the tax value of properties when properties are separated into their respective counties.


- Create a model for each county that had properties with a transaction.
