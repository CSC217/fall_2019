import numpy as np
import pandas as pd

def evaluate_linear_relationship(a, b):
    """

    Evaluate the linear relationship between two variables via least squares 
    and return the slope, intercept, predictions and residuals of the model.

    Parameters
    ----------

    a: list, numpy array or Pandas series
        The independent variable
    b: list, numpy array, or Pandas series
        The dependent variable

    Returns
    -------

    slope: float
        The slope of the linear relationship
    intercept: float
        The intercept of the linear relationship
    predictions: numpy array
        A list of predicted values for b given a
    residuals: numpy array
        The absolute difference between the predicted values and actual values
        of b


    slope:
    """
    assert (type(a) == list) or (type(a) == np.array) or (type(a) == pd.Series)
    assert (type(b) == list) or (type(b) == np.array) or (type(b) == pd.Series)
    if type(a) != np.array:
        a = np.array(a)
    if type(b) != np.array:
        b = np.array(b)
    slope = np.cov(a, b, bias=True)[0][1] / np.var(a)
    intercept = np.mean(b) - (slope * np.mean(a))
    predictions = (slope * a) + intercept
    residuals = b - predictions
    return slope, intercept, predictions, residuals


def pooled_variance(a, b):
    """

    Return the pooled variance of two distributions for a two-sided T test.

    Parameters
    ----------

    a: list, numpy array or Pandas series
        The first distribution
    b: list, numpy array, or Pandas series
        The second distribution

    Returns
    -------

    pooled_variance: float
        The pooled variance of the two variables.
    """
    return ((((len(a) - 1) * np.var(a, ddof=1)) + ((len(b) - 1) * np.var(b, ddof=1))) / (len(a) + len(b) - 2)) * ((1/len(a)) + 1/len(b))

def bootstrap_mean_diff(a, b, number_of_simulations=10000, seed_value=42):
    """

    Run a bootstrap simulation and return an array of simulated differences
    in the mean between two arrays (of a - b)

    Parameters
    ----------

    a: list, numpy array or Pandas series
        The first distribution
    b: list, numpy array, or Pandas series
        The second distribution
    number_of_simulations: int, optional
        The number of simulations to run. Default is 10,000.
    seed_value: int, optional
        The seed value to use for repeatable test results. Default is 42.

    Returns
    -------

    pooled_variance: float
        The pooled variance of the two variables.


    """
    np.random.seed(seed_value)
    diff = []
    for i in range(number_of_simulations):
        total = np.concatenate((a, b), axis=None)
        np.random.shuffle(total)
        new_a = total[:len(a)]
        new_b = total[len(a):]
        diff.append(np.mean(new_a) - np.mean(new_b))
    return diff
