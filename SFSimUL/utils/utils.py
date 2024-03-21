from typing import Hashable, List
import numpy as np 
from scipy.stats import boxcox, boxcox_llf


## Python versions of R's match() function 
# For each element of 'a', returns indices in 'b' that match a, or None if no match 
def match_list(a: List[Hashable], b: List[Hashable]) -> List[int]:
    return [b.index(x) if x in b else None for x in a]

def match(a: List[Hashable], b: List[Hashable]) -> List[int]:
    b_dict = {x: i for i, x in enumerate(b)}
    return [b_dict.get(x, None) for x in a]

## Function to shift values in array-like structures to be positive
# This is required for Box-Cox transforms
def shift_positive(x):
    if any(x <= 0): x = x - x.min() + 1e-4*(x.max() - x.min())
    return x 

## Perform a grid search for the best lambda for a Box-Cox transform 
def boxcox_gridsearch(x, lambda_min = -2.0, lambda_max = 2.0):
    # Convert input to numpy, ensure positive 
    orig_shape = x.shape 
    x = np.array(x).reshape(-1,)
    x = shift_positive(x)
    # Define grid of lambdas where the Box-Cox likelihood function is evaluated
    lambda_grid = np.linspace(lambda_min, lambda_max, num = 100)
    llf = np.zeros((100,))
    for idx in range(100):
        llf[idx] = boxcox_llf(lmb=lambda_grid[idx], data = x)
    # Extract argmax of likelihood 
    idxmax = np.argmax(llf)
    lammax = lambda_grid[idxmax]
    # Transform x with best lambda 
    y = boxcox(x=x, lmbda = lammax).reshape(orig_shape)
    return y, lammax

## "Flatten" (un-nest) a nested list 
def flatten_list(arg):
    if not isinstance(arg, list): # if not list
        return [arg]
    return [x for sub in arg for x in flatten_list(sub)] # recurse and collect


## Scale an array-like input to network range 
def scale_to_netrange(x):
    x = (x - x.min()) / (x.max() - x.min()) * (1 - -1) + (-1)
    return x


## Remove units from a variable name. 
# Units are bracketed [u] strings. This function drops them, if they exist. 
# E.g., 'Reff [pc]' becomes 'Reff'
def remove_units(s):
    if '[' in s and ']' in s:
        s = (s.split('[')[0] + s.split(']')[-1]).strip()
    return s