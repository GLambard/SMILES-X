import pandas as pd
import numpy as np
import math

from rdkit import Chem
# Disables RDKit whiny logging.
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

from sklearn.preprocessing import RobustScaler
from scipy.ndimage.interpolation import shift

np.set_printoptions(precision=3)

## SMILES RDKit checker 
# dataframe: provided dataframe with SMILES (column name: 'smiles') to check
# returns:
#         a dataframe with SMILES which passed the check
#         a list of SMILES which did not pass the check
def check_smiles(dataframe):
    smiles_veto = []
    bad_smiles_list = []
    for ismiles in dataframe.smiles:
        pass_tmp = True # True is an accepted molecule
        try:
            smi_tmp = Chem.MolToSmiles(Chem.MolFromSmiles(ismiles))
        except:
            pass_tmp = False
            bad_smiles_list.append(ismiles)
        smiles_veto.append(pass_tmp)
        
    dataframe = dataframe[smiles_veto].reset_index().iloc[:,1:]
    return dataframe, bad_smiles_list
##

## Split into train, valid, test sets, and standardize the targeted property (mean 0, std 1)
# smiles_input: array of SMILES to split
# prop_input: array of SMILES-associated property to split 
# train_index, valid_test_index: picked indices for training, validation and test 
# returns: 
#         3 arrays of smiles for training, validation, test: x_train, x_valid, x_test, 
#         3 arrays of properties for training, validation, test: y_train, y_valid, y_test, 
#         the scaling function: scaler
def split_standardize(smiles_input, prop_input, train_index, valid_test_index):
    
    x_train, x_valid_test = smiles_input[train_index], smiles_input[valid_test_index]
    y_train, y_valid_test = prop_input[train_index], prop_input[valid_test_index]
    valid_len = math.ceil(x_valid_test.shape[0]/2.)
    x_valid, x_test = x_valid_test[:valid_len], x_valid_test[valid_len:]
    y_valid, y_test = y_valid_test[:valid_len], y_valid_test[valid_len:]
    
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

## Step decay the learning rate during training
# initAlpha: initial learning rate 
# finalAlpha: final learning rate
# gamma: NewAlpha = initAlpha * (gamma ** exp), exp determined by the desired number of epochs
# epochs: desired number of epochs for training
class step_decay():
    def __init__(self, initAlpha = 1e-3, finalAlpha = 1e-5, gamma = 0.95, epochs = 100):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.finalAlpha = finalAlpha
        self.gamma = gamma
        self.epochs = epochs
        self.beta = (np.log(self.finalAlpha) - np.log(self.initAlpha)) / np.log(self.gamma)
        self.dropEvery = self.epochs / self.beta

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = epoch%self.dropEvery # epoch starts from 0, callbacks called from the beginning
        alpha = self.initAlpha * (self.gamma ** exp)

        # return the learning rate
        return float(alpha)
##