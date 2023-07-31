'''
import matplotlib.pyplot as plt

# define data
models = ['Model A', 'Model B', 'Model C']
scores = [0.8, 0.85, 0.9]

# create table plot
fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
ax.table(cellText=[scores], colLabels=models, loc='center')

# show plot
plt.show()

import pandas as pd

# Define a dictionary with model names and scores
models_dict = {'Model 1': 0.85,
               'Model 2': 0.92,
               'Model 3': 0.78}

# Convert the dictionary to a pandas dataframe
models_df = pd.DataFrame.from_dict(models_dict, orient='index', columns=['Score'])

# Sort the dataframe by score in descending order
models_df = models_df.sort_values(by='Score', ascending=False)

# Print the dataframe
print(models_df)

import pandas as pd
from time import sleep

linear_regression = {
    'Training set': {
        'MSE': 0.366826022801636,
        'R^2 score': 0.9991857817330744,
        'File created': 'y_train_pred_lin.txt'
    },
    'Testing set': {
        'MSE': 0.46479508466210306,
        'R^2 score': 0.9988275925304162,
        'File created': 'y_test_pred_lin.txt'
    },
    'Validation set': {
        'MSE': 0.5852152989143953,
        'R^2 score': 0.9984331134799277
    }
}

gradient_boosting = {
    'Training set': {
        'RMSE': 0.023013637765679166,
        'MSE': 0.0005296275232098943,
        'R^2 score': 0.9999988244225402,
        'File created': 'y_train_pred_gb.txt'
    },
    'Testing set': {
        'RMSE': 0.17778760320410078,
        'MSE': 0.03160843185305879,
        'R^2 score': 0.9999202703237852,
        'File created': 'y_test_pred_gb.txt'
    },
    'Validation set': {
        'RMSE': 0.47061446295699577,
        'MSE': 0.22147797274430156,
        'R^2 score': 0.9994070031138459
    }
}

# Convert the dictionaries to DataFrames
linear_regression_df = pd.DataFrame.from_dict(linear_regression, orient='index')
gradient_boosting_df = pd.DataFrame.from_dict(gradient_boosting, orient='index')

# Add a 'Model' column to the DataFrames
linear_regression_df['Model'] = 'Linear Regression Model'
gradient_boosting_df['Model'] = 'Gradient Boosting Model'

# Set the index
linear_regression_df.set_index(['Model', linear_regression_df.index], inplace=True)
gradient_boosting_df.set_index(['Model', gradient_boosting_df.index], inplace=True)

# Concatenate the DataFrames
df = pd.concat([linear_regression_df, gradient_boosting_df])

# Rename the index levels
df.index.names = ['Model', 'Set']

# Display the DataFrame
print(df)
sleep(10)
'''
import pandas as pd
import matplotlib.pyplot as plt

# Create the dataframe
data = {'Model': ['Linear Regression', 'Gradient Boosting'],
        'Set': ['Training set', 'Testing set', 'Validation set']*2,
        'Metric': ['MSE', 'R^2 score', 'File created']*3,
        'Value': [0.366826022801636, 0.9991857817330744, 'y_train_pred_lin.txt',
                  0.46479508466210306, 0.9988275925304162, 'y_test_pred_lin.txt',
                  0.5852152989143953, 0.9984331134799277, '-',
                  0.0005296275232098943, 0.9999988244225402, 'y_train_pred_gb.txt',
                  0.03160843185305879, 0.9999202703237852, 'y_test_pred_gb.txt',
                  0.22147797274430156, 0.9994070031138459, '-']}

df = pd.DataFrame(data)
df = df.pivot(index=['Model', 'Set'], columns='Metric', values='Value')

# Plot the table
fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')

# Save the image
plt.savefig('table.png')

