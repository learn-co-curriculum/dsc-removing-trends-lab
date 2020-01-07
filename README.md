
# Removing Trends - Lab

## Introduction

In this lab, you'll practice your detrending skills!

## Objectives

In this lab you will: 

- Use a log transformation to minimize non-stationarity 
- Use rolling means to reduce non-stationarity 
- Use differencing to reduce non-stationarity 
- Use rolling statistics as a check for stationarity 
- Create visualizations of transformed time series as a visual aid to determine if stationarity has been achieved 
- Use the Dickey-Fuller test and conclude whether or not a dataset is exhibiting stationarity 


## Detrending the Air passenger data 

In this lab you will work with the air passenger dataset available in `'passengers.csv'`. First, run the following cell to import the necessary libraries. 


```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
```

- Import the `'passengers.csv'` dataset 
- Change the data type of the `'Month'` column to a proper date format 
- Set the `'Month'` column as the index of the DataFrame 
- Print the first five rows of the dataset 


```python
# Import 'passengers.csv' dataset
data = None

# Change the data type of the 'Month' column
data['Month'] = None

# Set the 'Month' column as the index
ts = None

# Print the first five rows

```

Plot this time series. 


```python
# Plot the time series

```

## Create a stationarity check

Your next task is to use the code from previous labs to create a function `stationarity_check()` that takes in a time series and performs stationarity checks including rolling statistics and the Dickey-Fuller test. 

We want the output of the function to: 

- Plot the original time series along with the rolling mean and rolling standard deviation (use a window of 8) in one plot 
- Output the results of the Dickey-Fuller test 


```python
# Create a function to check for the stationarity of a given time series using rolling stats and DF test
# Collect and package the code from previous labs

```

Use your newly created function on the `ts` timeseries. 


```python
# Code here
```

## Perform a log and square root transform

Plot a log transform of the original time series (`ts`). 


```python
# Plot a log transform

```

Plot a square root  transform of the original time series (`ts`). 


```python
# Plot a square root transform

```

Going forward, let's keep working with the log transformed data before subtracting rolling mean, differencing, etc.

## Subtracting the rolling mean

Create a rolling mean using your log transformed time series, with a time window of 7. Plot the log-transformed time series and the rolling mean together.


```python
# your code here
roll_mean = None
fig = plt.figure(figsize=(11,7)) 

```

Now, subtract this rolling mean from the log transformed time series, and look at the 10 first elements of the result.  


```python
# Subtract the moving average from the log transformed data
data_minus_roll_mean = None

# Print the first 10 rows

```

Drop the missing values from this time series. 


```python
# Drop the missing values

```

Plot this time series now. 


```python
# Plot the result

```

Finally, use your function `check_stationarity()` to see if this series is stationary!


```python
# Your code here
```

### Based on the visuals and on the Dickey-Fuller test, what do you conclude?



```python
# Your conclusion here
```

## Subtracting the weighted rolling mean

Repeat all the above steps to calculate the exponential *weighted* rolling mean with a halflife of 4. Start from the log-transformed data again. Compare the Dickey-Fuller test results. What do you conclude?


```python
# Calculate Weighted Moving Average of log transformed data
exp_roll_mean = None

# Plot the original data with exp weighted average

```

- Subtract this exponential weighted rolling mean from the log transformed data  
- Print the resulting time series 


```python
# Subtract the exponential weighted rolling mean from the original data 
data_minus_exp_roll_mean = None

# Plot the time series

```

Check for stationarity of `data_minus_exp_roll_mean` using your function. 


```python
# Do a stationarity check
```

### Based on the visuals and on the Dickey-Fuller test, what do you conclude?



```python
# Your conclusion here
```

## Differencing

Using exponentially weighted moving averages, we seem to have removed the upward trend, but not the seasonality issue. Now use differencing to remove seasonality. Make sure you use the right amount of `periods`. Start from the log-transformed, exponentially weighted rolling mean-subtracted series.

After you differenced the series, drop the missing values, plot the resulting time series, and then run the `stationarity check()` again.


```python
# Difference your data
data_diff = None

# Drop the missing values


# Check out the first few rows
data_diff.head(15)
```

Plot the resulting differenced time series. 


```python
# Plot your differenced time series

```


```python
# Perform the stationarity check
```

### Your conclusion


```python
# Your conclusion here
```

## Summary 

In this lab, you learned how to make time series stationary through using log transforms, rolling means, and differencing.
