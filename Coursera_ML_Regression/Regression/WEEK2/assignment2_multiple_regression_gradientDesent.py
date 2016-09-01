# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:30:49 2016

@author: nabbassi
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from math import sqrt


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


# funtion for the conversion of the data into numpy matrix
def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.as_matrix()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    #output_sarray = pd.DataFrame(output)
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output.as_matrix()
    return(feature_matrix, output_array)
#function to predict the valus from the features and weights
def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix,weights)

    return(predictions)  
#funtion for the derivative for gradient descent
def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative =  2*np.dot(feature,errors)
    return(derivative)  

#implementation of regression gradient desent algorithms

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix,weights)
        
        # compute the errors as predictions - output
        errors = predictions - output

        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            
            derivative = feature_derivative(feature_matrix[:,i],errors)
            

            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares = gradient_sum_squares + derivative*derivative
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - step_size*derivative
            
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)    


################# REGRESSION USING GB for 1 feature ##################################################

# let's test out the gradient descent
simple_features = ['sqft_living']

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, train_csv.price)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7


simple_reg_weights = regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,tolerance)

(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, test_csv.price)
model1_predicted_output = predict_output(test_simple_feature_matrix,simple_reg_weights)



error_model1 = model1_predicted_output - test_output
RSS = sum(error_model1*error_model1)
print "------------------------------------------------------------------------------------------------------"
print "------------------------------------- Linear Regression using Gradient Descent------------------------"
print "------------------------------------------------------------------------------------------------------"
print "The learned weights for the single feature 'sqft_living' through GD are: " + str(simple_reg_weights)
print "The predicted price for the first element is: " + str(model1_predicted_output[0])

# Calculating RSS of the model

error_model1 = model1_predicted_output - test_output
RSS = sum(error_model1*error_model1)
print "RSS of the simple Regression using GD is: " + str(RSS)
print "-------------------------------------------------------------------------------------------------------"

########################################################################################################

#################################### RUNNING A MULTIPLE REGRESSION #####################################


model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, train_csv.price)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

multi_reg_weights = regression_gradient_descent(feature_matrix,output,initial_weights,step_size,tolerance)
(test_multi_feature_matrix, test_output) = get_numpy_data(test_data,model_features, test_csv.price)
model2_predicted_output = predict_output(test_multi_feature_matrix,multi_reg_weights)


print "------------------------------------------------------------------------------------------------------"
print "------------------------------------- Multiple Regression using Gradient Descent------------------------"
print "------------------------------------------------------------------------------------------------------"
print "The learned weights for the features 'sqft_living and sqft_living15' through GB are: " + str(multi_reg_weights)
print "The predicted price for the first element is: " + str(model1_predicted_output[0])

# Calculating RSS of the model

error_model1 = model1_predicted_output - test_output
RSS = sum(error_model1*error_model1)
print "RSS of the multiple Regression using GB is: " + str(RSS)
print "-------------------------------------------------------------------------------------------------------"