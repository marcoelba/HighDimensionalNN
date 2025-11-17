import numpy as np


def sum_not_nan(x):
    x_shape = x.shape
    return (~np.isnan(x)).sum(axis=-1) if x.shape[-1] > 1 else (~np.isnan(x)).sum(axis=-1).sum(axis=-1)

def get_indeces(dict_arrays):
    where_not_na = [sum_not_nan(item) > 0 for key, item in dict_arrays.items()]

    where_all = np.prod(np.array(where_not_na), axis=0)
    print("\n Total not NAs: ", where_all.sum())
    
    return where_all
