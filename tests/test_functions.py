from resources import *
import pandas as pd
import pytest


def test_moving_average():
    # Test case 1
    df = pd.DataFrame({
        'Close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    try:
        Functions.moving_average(df)
    except:
        assert False

    assert True