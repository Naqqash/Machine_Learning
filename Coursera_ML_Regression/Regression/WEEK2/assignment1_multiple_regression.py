# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:53:09 2016

@author: nabbassi
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from math import log


# Making sure that the features have proper data format
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

# Reading the csv files using panda
data_csv = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)
train_csv = pd.read_csv('kc_house_train_data.csv',dtype=dtype_dict)
test_csv = pd.read_csv('kc_house_test_data.csv',dtype=dtype_dict)

# Conversting the csv data to the dataframe and deleting the extra columns which are not relevant

house_train = pd.DataFrame(train_csv)
train_data = house_train.drop(['price','id','date'],axis=1)
house_test = pd.DataFrame(test_csv)
test_data  = house_test.drop(['price','id','date'],axis=1)


# create some features as an example to create example model
example_features = ['sqft_living', 'bedrooms', 'bathrooms']


# Creating the example model using the linearRegression class
example_model = LinearRegression()
#fitting the model with the example features data and the target is 'price'
example_model.fit(train_csv[example_features],train_csv.price)

#print the coeffecient of the learned model

#print pd.DataFrame(zip(train_csv[example_features],example_model.coef_),columns=['features','estimatedCoeffecients'])

""" Function to calculate the RSS 
input --> model object, data and the target
output --> RSS
"""

def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)

    # Then compute the residuals/errors
    error = outcome - predictions
    

    # Then square and add them up
    square_errors = error * error
    
    RSS = square_errors.sum()

    return(RSS)    

# Calculating the RSS of the example model    
rss_example_train = get_residual_sum_of_squares(example_model, test_csv[example_features], test_csv['price'])
#print rss_example_train # should be 2.7376153833e+14
    



# Creating new features

# creat a feature named "bedrooms_squared"

train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)

# create a feature named "bed_bath_rooms"

train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']


#create a feature named "log_sqft_living"

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))

#create lattitude + logitude feature

train_data['lat_plus_long'] = train_data['lat'] + train_data['long']
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

# creating features for different models 

model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long','bed_bath_rooms']
model_3_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long','bed_bath_rooms','bedrooms_squared','log_sqft_living','lat_plus_long']


# Creating the model1 using the linearRegression class
model1 = LinearRegression()
#fitting the model1 with the model_1_features data and the target is 'price'
model1.fit(train_data[model_1_features],train_csv.price)


# Creating the model2 using the linearRegression class
model2 = LinearRegression()
#fitting the model1 with the model_2_features data and the target is 'price'
model2.fit(train_data[model_2_features],train_csv.price)
#
#
# Creating the model3 using the linearRegression class
model3 = LinearRegression()
#fitting the model1 with the model_1_features data and the target is 'price'
model3.fit(train_data[model_3_features],train_csv.price)

#Coffecients of Model1
print "----------------------------------------------MODEL1---------------------------------------"
print pd.DataFrame(zip(train_data[model_1_features],model1.coef_),columns=['features','estimatedCoeffecients'])

#Coffecients of Model2
print "----------------------------------------------MODEL2---------------------------------------"
print pd.DataFrame(zip(train_data[model_2_features],model2.coef_),columns=['features','estimatedCoeffecients'])

#Coffecients of Model3
print "----------------------------------------------MODEL3---------------------------------------"
print pd.DataFrame(zip(train_data[model_3_features],model3.coef_),columns=['features','estimatedCoeffecients'])

print "------------------------------------------------------------------------------------------"

# RSS of the models on the train data

print "----------------------RSS of the models on TRAIN DATA-------------------------------"
print "------------------------------------------------------------------------------------------"

rss_model1_train = get_residual_sum_of_squares(model1, train_data[model_1_features], train_csv['price'])
rss_model2_train = get_residual_sum_of_squares(model2, train_data[model_2_features], train_csv['price'])
rss_model3_train = get_residual_sum_of_squares(model3, train_data[model_3_features], train_csv['price'])

print "RSS MODEL1 ON TRAIN DATA ----> " + str(rss_model1_train)
print "RSS MODEL2 ON TRAIN DATA----> " + str(rss_model2_train)
print "RSS MODEL3 ON TRAIN DATA----> " + str(rss_model3_train)


print "------------------------------------------------------------------------------------------"

print "----------------------RSS of the models on TEST DATA-------------------------------"
print "------------------------------------------------------------------------------------------"

rss_model1_test = get_residual_sum_of_squares(model1, test_data[model_1_features], test_csv['price'])
rss_model2_test = get_residual_sum_of_squares(model2, test_data[model_2_features], test_csv['price'])
rss_model3_test = get_residual_sum_of_squares(model3, test_data[model_3_features], test_csv['price'])

print "RSS MODEL1 ON TEST DATA----> " + str(rss_model1_test)
print "RSS MODEL2 ON TEST DATA----> " + str(rss_model2_test)
print "RSS MODEL3 ON TEST DATA----> " + str(rss_model3_test)


print "------------------------------------------------------------------------------------------"

