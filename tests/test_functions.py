import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import resources

from resources import functions
import pandas as pd


def test_moving_average():
    # Test case 1
    df = pd.DataFrame({
        'Close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    try:
        functions.moving_average(df)
    except:
        assert False

    assert True