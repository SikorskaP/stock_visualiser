import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import resources

from resources import functions
import pandas as pd
import numpy as np
import builtins
import ta
import matplotlib.pyplot as plt


def test_pick_periods(monkeypatch):
    # Test case inputs
    inputs = ['14', '3']

    # Monkeypatch the input() function to return the test case inputs
    monkeypatch.setattr(builtins, 'input', lambda x: inputs.pop(0))
    
    # Call the pick_periods function
    k_period, d_period = functions.Functions.pick_periods()
    
    # Check the output
    assert k_period == 14
    assert d_period == 3

def test_rsi_macd():
    # Create a test dataframe
    test_df = pd.DataFrame({
        'Open': np.random.randint(1, 100, 200),
        'High': np.random.randint(1, 100, 200),
        'Low': np.random.randint(1, 100, 200),
        'Close': np.random.randint(1, 100, 200),
        'Volume': np.random.randint(1, 100000, 200)
         })


    # Call the rsi_macd function
    print(test_df)
    test_df['test_RSI'] = ta.momentum.rsi(test_df.Close,window=14)
    functions.Functions.rsi_macd(test_df)

    print(test_df.head(30))
    # Check that the dataframe has the expected columns
    pd.testing.assert_series_equal(test_df['RSI'], test_df['test_RSI'])

    # Check that the plot function was called
    # assert plt.gcf().get_label() == 'MACD + RSI'
                         
