import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import RobustScaler
from scipy.ndimage.interpolation import shift

np.set_printoptions(precision=3)

## Train/Validation/Test random split
# smiles_input: array of SMILES to split
# prop_input: array of SMILES-associated property to split 
# random state: random seed for reproducibility
# scaling: property scaling (mean = 0, s.d. = 1) (default: True)
# returns: 
#         3 arrays of smiles for training, validation, test: x_train, x_valid, x_test, 
#         3 arrays of properties for training, validation, test: y_train, y_valid, y_test, 
#         the scaling function: scaler
def random_split(smiles_input, prop_input, random_state, scaling = True):

    full_idx = np.array([x for x in range(smiles_input.shape[0])])
    train_idx = np.random.choice(full_idx, 
                                 size=math.ceil(0.8*smiles_input.shape[0]), 
                                 replace = False)
    x_train = smiles_input[train_idx]
    y_train = prop_input[train_idx].reshape(-1, 1)
    
    valid_test_idx = full_idx[np.isin(full_idx, train_idx, invert=True)]
    valid_test_len = math.ceil(0.5*valid_test_idx.shape[0])
    valid_idx = valid_test_idx[:valid_test_len]
    test_idx = valid_test_idx[valid_test_len:]
    x_valid = smiles_input[valid_idx]
    y_valid = prop_input[valid_idx].reshape(-1, 1)
    x_test = smiles_input[test_idx]
    y_test = prop_input[test_idx].reshape(-1, 1)
    
    if scaling == True:
        scaler = RobustScaler(with_centering=True, 
                              with_scaling=True, 
                              quantile_range=(5.0, 95.0), 
                              copy=True)
        scaler_fit = scaler.fit(y_train)
        print("Scaler: {}".format(scaler_fit))
        y_train = scaler.transform(y_train)
        y_valid = scaler.transform(y_valid)
        y_test = scaler.transform(y_test)
    
    print("Train/valid/test splits: {0:0.2f}/{1:0.2f}/{2:0.2f}\n\n".format(\
                                      x_train.shape[0]/smiles_input.shape[0],\
                                      x_valid.shape[0]/smiles_input.shape[0],\
                                      x_test.shape[0]/smiles_input.shape[0]))
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test, scaler
##

## Compute mean and median of predictions
# x_cardinal_tmp: number of augmented SMILES for each original SMILES
# y_pred_tmp: predictions to be averaged
# returns: 
#         arrays of mean and median predictions
def mean_median_result(x_cardinal_tmp, y_pred_tmp):
    x_card_cumsum = np.cumsum(x_cardinal_tmp)
    x_card_cumsum_shift = shift(x_card_cumsum, 1, cval=0)

    y_mean = \
    np.array(\
             [np.mean(y_pred_tmp[x_card_cumsum_shift[cenumcard]:ienumcard]) \
              for cenumcard,ienumcard in enumerate(x_card_cumsum.tolist())]\
            )

    y_med = \
    np.array(\
             [np.median(y_pred_tmp[x_card_cumsum_shift[cenumcard]:ienumcard]) \
              for cenumcard,ienumcard in enumerate(x_card_cumsum.tolist())]\
            )
    
    return y_mean, y_med
##
