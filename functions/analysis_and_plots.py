import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error









def preprocessing(df):
        
        """
        Perform data preprocessing tasks on a stacked temperature DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame with temperature data in a stacked format.

        Returns:
        - DataFrame: Preprocessed DataFrame with columns 'Date' and 'Temp'.

        Example:
        preprocessed_df = preprocessing(input_df)
        """

        df_1 = df.copy()
        df_1.drop(['D-J-F', 'M-A-M', 'J-J-A', 'S-O-N', 'Temp'], axis=1, inplace=True)
        df_1.rename(columns={"JAN":1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6,
                        'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}, inplace=True)
        df_1.set_index('Year', inplace=True)
        stacked_df = df_1.stack()
        stacked_df = stacked_df.reset_index()

        stacked_df.columns = ['Year', 'Month', 'Temp']
        stacked_df['Date'] = stacked_df['Year'].astype(str) + '-' + stacked_df['Month'].astype(str)
        df_main = stacked_df.copy()
        df_main.drop(columns=['Year', 'Month'], inplace=True)
        df_main = df_main[['Date', 'Temp']]

        return df_main


def train_test_split(df, split_index=455):
    """
    Splits a DataFrame into training and test sets.

    Parameters:
        df (DataFrame): The input DataFrame to be split.
        split_index (int): The index at which to split the DataFrame. Default is 455.

    Returns:
        train_data (DataFrame): The training data containing rows up to index (split_index-1).
        test_data (DataFrame): The test data containing rows from index split_index onwards.

    Example:
    Suppose you have a DataFrame df containing temperature data and you want to split it into training and test sets at index 400:
    
    train_data, test_data = train_test_split(df, split_index=400)
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    """
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid value for 'domain'. It should be within the range of the DataFrame.")
    
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    
    return train_data, test_data

def plot_train_test_data(train_data, test_data, x_label_index=50,
                         figsize=(12,5), dpi=150):
    
    # Concatenate train_data and test_data
    combined_data = pd.concat([train_data, test_data])

    # Create a new index for displaying every 100 values
    xlabels = combined_data.index
    xlabels = xlabels[::x_label_index]

    # Set the style for the Seaborn plot
    sns.set(style="whitegrid")

    # Create the Seaborn plot
    plt.figure(figsize=figsize, dpi=dpi)
    sns.lineplot(data=combined_data, x=combined_data.Date, y='Temp', label='TRAIN', legend=True)

    sns.lineplot(data=test_data, x=test_data.Date, y='Temp', label='TEST', legend=True)

    # Set the x-axis labels
    plt.xticks(xlabels)

    # Add labels and a title (customize as needed)
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Temperature Data')

    # Show the plot
    plt.legend()
    plt.tight_layout()
    plt.show()






def seasonal_decompose_plot(train_data, period=12):
    
    """
    Perform seasonal decomposition of temperature data and create a plot of the decomposition components.

    Parameters:
        train_data (DataFrame): The training data containing temperature observations.
        period (int, optional): The seasonal period. Default is 12 (assuming monthly data).

    Returns:
        None

    Example:
    Suppose you have a DataFrame `train_data` containing temperature data, and you want to perform seasonal decomposition with a period of 12 (assuming monthly data). You can call the function as follows:

    seasonal_decompose_plot(train_data, period=12)

    This will generate a plot showing the observed, trend, seasonal, and residual components of the temperature data.
    """
    
    # Perform seasonal decomposition on train_data
    result = seasonal_decompose(train_data['Temp'], period=period)
    
    # Create the decomposition plot
    plt.figure(figsize=(8, 6))

    # Plot the observed component
    plt.subplot(411)
    plt.plot(result.observed, label='Observed')
    plt.legend()

    # Plot the trend component
    plt.subplot(412)
    plt.plot(result.trend, label='Trend')
    plt.legend()

    # Plot the seasonal component
    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend()

    # Plot the residual component
    plt.subplot(414)
    plt.plot(result.resid, label='Residual')
    plt.legend()

    # Set the x-axis labels
    xlabels = train_data.index
    xlabels = xlabels[::100]
    xlabels = xlabels.append(pd.Index([xlabels[-1]]))
    plt.xticks(xlabels)

    # Add labels and a title (customize as needed)
    plt.xlabel('Date')
    plt.suptitle('Seasonal Decomposition of Temperature Data')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the subplot layout to make room for the suptitle
    plt.savefig('seasonal_decomposition_plot.png', dpi=300, bbox_inches='tight') 
    # Show the plot
    plt.show()

    
def exp_smothing_model(train_data, test_data, model='triple', span=12):
    """
    Perform simple exponential smoothing and generate temperature predictions.

    Parameters:
        train_data (DataFrame): Training data containing historical temperature observations.
        test_data (DataFrame): Test data containing dates for forecasting.
        model (str, optional): Smoothing model type ('single', 'double', or 'triple'). Default is 'triple'.
        span (int, optional): Span for single exponential smoothing. Default is 12.
        time_steps (int): represents the number of time steps into the future you want to make predictions for. and
                            must be equvalent length of test_data df

    Returns:
        DataFrame: DataFrame containing date and temperature predictions.

    Example:
    Suppose you have training data `train_data` and test data `test_data` containing temperature observations and dates for forecasting. To use the 'triple' exponential smoothing model with a span of 12 for single exponential smoothing, you can call the function as follows:

    predictions = simple_exp_smothing_model(train_data, test_data, model='triple', span=12)

    This will return a DataFrame `predictions` containing forecasted temperatures with corresponding dates.
    """
    time_steps = len(test_data)
    
    if model == 'single':
        span = span # The model will consider the last 12 months weighted average for forecasting
        alpha = 2/(span+1)
        model_fit = SimpleExpSmoothing(train_data['Temp']).fit(alpha)
    
    elif model == 'double':
        model_fit = ExponentialSmoothing(train_data['Temp'],trend='add').fit()
    
    elif model == 'triple':
        model_fit = ExponentialSmoothing(train_data['Temp'],trend='add',seasonal='add',seasonal_periods=12).fit()
        
   
    test_predictions = model_fit.forecast(time_steps).rename('Temp')
    predictions = pd.DataFrame(test_predictions)
    predictions['Date'] = test_data['Date']
    predictions = predictions[['Date', 'Temp']]
    return predictions
    
       
def exponential_smothing_plot(train_data, test_data, predictions, split_index=100,
                              figsize=(12, 6), dpi=150):
    """
    Create a line plot to visualize temperature data, including training data, test data, and predictions.

    Parameters:
    - train_data (DataFrame): DataFrame containing training data with columns 'Date' and 'Temp'.
    - test_data (DataFrame): DataFrame containing test data with columns 'Date' and 'Temp'.
    - predictions (DataFrame): DataFrame containing predictions with columns 'Date' and 'Temp'.
    - split_index (int, optional): Frequency for displaying x-axis labels. Defaults to 100.
    - figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (12, 6).
    - dpi (int, optional): Dots per inch for the plot. Defaults to 150.

    Returns:
    - None: Displays the plot.

    Example:
    exponential_smothing_plot(train_df, test_df, predictions_df)
    """
    
    # Concatenate train_data and test_data
    combined_data = pd.concat([train_data, test_data])

    # Create a new index for displaying every 100 values
    xlabels = combined_data.index
    xlabels = xlabels[::split_index]
    
    # Set the style for the Seaborn plot (optional)
    sns.set(style="whitegrid")

    # Create the Seaborn plot
    plt.figure(figsize=figsize, dpi=dpi)
    sns.lineplot(data=combined_data, x=combined_data.Date, y='Temp', label='TRAIN', legend=True)
    sns.lineplot(data=predictions, x=predictions.Date, y='Temp', label='PREDICTION', legend=True)
    sns.lineplot(data=test_data, x=test_data.Date, y='Temp', label='TEST', legend=True)

    # Set the x-axis labels
    plt.xticks(xlabels)

    # Add labels and a title (customize as needed)
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Temperature Data')

    # Show the plot
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    
def rms_error_calc(test_data, predictions):
        
        """
        Calculate the Root Mean Square Error (RMSE) between observed and predicted temperature data.

        Parameters:
        - test_data (DataFrame): DataFrame containing observed (test) temperature data with a 'Temp' column.
        - predictions (DataFrame): DataFrame containing predicted temperature data with a 'Temp' column.

        Returns:
        - float: The RMSE value.

        Example:
        rmse = rms_error_calc(test_df, predictions_df)
        """
    
        rms_error = np.sqrt(mean_squared_error(test_data['Temp'], predictions['Temp']))
        
        return rms_error
        
        
        
        
