# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:45:10 2016

@author: nabbassi
"""
import pandas as pd

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
data_csv = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)

train_csv = pd.read_csv('kc_house_train_data.csv',dtype=dtype_dict)

test_csv = pd.read_csv('kc_house_test_data.csv',dtype=dtype_dict)




def simple_linear_regression(input_feature, output):
    
    length = input_feature.size
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
    numerator = sum_product_X_Y - (sum_feature*sum_output)/length
    denominator = input_feature_square_sum - (sum_feature*sum_feature)/length
    slope = numerator/denominator
    
    # use the formula for the intercept
    intercept = output.mean()-slope*(input_feature.mean())
    
    return (intercept, slope)
    
def get_regression_predictions(input_feature, intercept, slope):
    
    predicted_output = intercept + slope*input_feature
        
    return(predicted_output)

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predicted_values = intercept + slope*input_feature 

    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    difference = predicted_values - output

    # square the residuals and add them up
    RSS = sum(difference * difference)

    return(RSS)

def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output-intercept)/slope

    return estimated_feature


sqft_intercept, sqft_slope = simple_linear_regression(train_csv.sqft_living, train_csv.price)
print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)

test_house_sqft = 2650

estimated_price = get_regression_predictions(test_house_sqft,sqft_intercept,sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (test_house_sqft, estimated_price)

rss_prices_on_sqft = get_residual_sum_of_squares(train_csv.sqft_living, train_csv.price, sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)


my_house_price = 700074.85
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)



# Estimate the slope and intercept for predicting 'price' based on 'bedrooms'

bedroom_intercept, bedroom_slope = simple_linear_regression(train_csv.bedrooms, train_csv.price)

print "Intercept: " + str(bedroom_intercept)
print "Slope: " + str(bedroom_slope)

# Compute RSS when using bedrooms on TEST data:
rss_prices_on_sqft = get_residual_sum_of_squares(test_csv.bedrooms, test_csv.price, bedroom_intercept, bedroom_slope)
print rss_prices_on_sqft

# Compute RSS when using squarefeet on TEST data:
rss_prices_on_sqft = get_residual_sum_of_squares(test_csv.sqft_living, test_csv.price, sqft_intercept, sqft_slope)
print rss_prices_on_sqft






