# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:01:21 2016

@author: nabbassi
"""

import sframe
sales = sframe.SFrame('kc_house_data.gl/')
train_data,test_data = sales.random_split(.8,seed=0)

def simple_linear_regression(input_feature, output):
    
    N = input_feature.size()
    # compute the sum of input_feature and output
    
    sum_feature = input_feature.sum()
    sum_output  = output.sum()
    
    # compute the product of the output and the input_feature and its sum
    product_X_Y= input_feature*output
    sum_product_X_Y = product_X_Y.sum()
    
    # compute the squared value of the input_feature and its sum
    input_feature_square = input_feature*input_feature
    input_feature_square_sum = input_feature_square.sum()
    
    # use the formula for the slope
    numerator = sum_product_X_Y - (sum_feature*sum_output)/N
    denominator = input_feature_square_sum - (sum_feature*sum_feature)/N
    slope = numerator/denominator
    
    # use the formula for the intercept
    intercept = output.mean()-slope*(input_feature.mean())
    
    return (intercept, slope)
    
def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + slope*input_feature
    
    return predicted_values



def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predicted_values = intercept + slope*input_feature 

    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    difference = predicted_values - output

    # square the residuals and add them up
    RSS = difference * difference

    return(RSS)
    
    
    
    
    
sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)

