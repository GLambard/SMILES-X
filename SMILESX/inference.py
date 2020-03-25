import numpy as np
import pandas as pd
import os
import glob

from rdkit import Chem

from tensorflow.keras.models import load_model
from tensorflow.keras import metrics

from SMILESX import utils, model, token, augm, main

from pickle import load

## Inference on the SMILESX predictions
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for inference (Default: None, i.e. automatically detect k_fold_number from main.Main phase)
# augmentation: SMILES's augmentation (Default: False)
# indir: directory of already trained prediction models (*.hdf5) and vocabulary (*.txt) (Default: '../data/')
# outdir: directory for outputs (plots + .txt files) -> 'Inference/'+'{}/{}/'.format(data_name,p_dir_temp) is then created (Default: '../data/')
# n_gpus: number of GPUs to be used in parallel (Default: 1)
# gpus_list: list of GPU IDs to be used (Default: None), e.g. ['0','1','2']
# gpus_debug: print out the GPUs ongoing usage 
# returns:
#         Array of SMILES with their inferred property (mean, standard deviation) from models ensembling
class Inference:

    def __init__(self, 
                 data_name, 
                 data_units = '',
                 k_fold_number = None,
                 augmentation = False, 
                 indir = "../data/", 
                 outdir = "../data/", 
                 n_gpus = 1, 
                 gpus_list = None, 
                 gpus_debug = False):
        
        self.data_name = data_name
        self.data_units = data_units
        self.k_fold_number = k_fold_number
        self.augmentation = augmentation
        
        # GPUs options
        self.strategy, self.gpus = main.set_gpuoptions(n_gpus = n_gpus, 
                                                       gpus_list = gpus_list, 
                                                       gpus_debug = gpus_debug)
        if self.strategy is None:
            return
        ##
        
        if augmentation:
            p_dir_temp = 'Augm'
            self.canonical = False
            self.rotation = True
            print("Data augmentation is required.")
        else:
            p_dir_temp = 'Can'
            self.canonical = True
            self.rotation = False
            print("No data augmentation is required.")

        self.input_dir = indir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
        self.save_dir = outdir+'Inference/'+'{}/{}/'.format(data_name,p_dir_temp)
        os.makedirs(self.save_dir, exist_ok=True)

        for itype in ["txt","hdf5","pkl"]:
            exists_file = glob.glob(self.input_dir + "*." + itype)
            exists_file_len = len(exists_file)
            if exists_file_len > 0:
                if itype == "hdf5":
                    if self.k_fold_number is None:
                        self.k_fold_number = exists_file_len
            else:
                print("***Process of inference automatically aborted!***")
                if itype == "hdf5":
                    print("The input directory does not contain any trained models (*.hdf5 files).\n")
                else:
                    print("The input directory does not contain any vocabulary (*_Vocabulary.txt file) or data scaler (*.pkl file).\n")
                return
        
        # Setting up the scalers, trained models, and vocabulary
        self.scalers_list = []
        self.models_list = []
        self.max_length = 0
        for ifold in range(self.k_fold_number):
            # Load the scaler
            self.scalers_list.append(load(open(self.input_dir+'scaler_fold_' + str(ifold) + '.pkl', 'rb')))

            # Model's architecture
            self.models_list.append(load_model(self.input_dir+'LSTMAtt_'+self.data_name+'_model.best_fold_'+str(ifold)+'.hdf5', 
                                    custom_objects={'AttentionM': model.AttentionM()}))
            
            # max_length retrieval
            if ifold == 0:
                # Maximum of length of SMILES to process
                self.max_length = self.models_list[0].layers[0].output_shape[-1][1]
                print("Maximum length of tokenized SMILES: {} tokens.".format(self.max_length))
                
        # Tokens as a list
        tokens = token.get_vocab(self.input_dir+self.data_name+'_Vocabulary.txt')
        # Add 'pad', 'unk' tokens to the existing list
        vocab_size = len(tokens)
        self.tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)
        print("Full vocabulary: {}, of size: {}.\n".format(self.tokens, vocab_size))
        
        print("***************************************")
        print("***SMILES_X for inference initiated.***")
        print("***************************************\n")

    # smiles_list: targeted SMILES list for property inference (Default: ['CC','CCC','C=O'])
    # check_smiles: check the SMILES' correctness via RDKit (Default: True)
    def infer(self, smiles_list = ['CC','CCC','C=O'], check_smiles = True):

        print("**************************************")
        print("***SMILES_X for inference starts...***")
        print("**************************************\n")

        if check_smiles:
            print("Checking the SMILES list for inference.")
            smiles_checked = list()
            smiles_rejected = list()
            for ismiles in smiles_list:
                mol_tmp = Chem.MolFromSmiles(ismiles)
                if mol_tmp != None:
                    smiles_can = Chem.MolToSmiles(mol_tmp)
                    smiles_checked.append(smiles_can)
                else:
                    smiles_rejected.append(ismiles)

            if len(smiles_rejected) > 0:
                with open(self.save_dir+'rejected_smiles.txt','w') as f:
                    for ismiles in smiles_rejected:
                        f.write("%s\n" % ismiles)
                print("Check the {} file for {} rejected SMILES.".format(self.save_dir+'rejected_smiles.txt', len(smiles_rejected)))

            if len(smiles_checked) == 0:
                print("***Process of inference automatically aborted!***")
                print("The provided SMILES are all incorrect and could not be sanitized via RDKit.\n")
                return
        else:
            smiles_checked = smiles_list

        smiles_x = np.array(smiles_checked)
        smiles_y = np.array([[np.nan]*len(smiles_checked)]).flatten()

        smiles_x_enum, smiles_x_enum_card, smiles_y_enum = \
        augm.Augmentation(smiles_x, smiles_y, canon=self.canonical, rotate=self.rotation)

        print("Number of enumerated SMILES: {}.".format(smiles_x_enum.shape[0]))

        print("Tokenization of SMILES.\n")
        # Tokenize SMILES 
        smiles_x_enum_tokens = token.get_tokens(smiles_x_enum)

        # Encode the tokens to integers
        smiles_x_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = smiles_x_enum_tokens, 
                                                             max_length = self.max_length, 
                                                             vocab = self.tokens)
        
        # models ensembling
        smiles_y_pred_mean_array = np.empty(shape=(0,len(smiles_checked)), dtype='float')
        for ifold in range(self.k_fold_number):
                
            # predict and compare for the training, validation and test sets
            smiles_y_pred = self.models_list[ifold].predict(smiles_x_enum_tokens_tointvec)

            # compute a mean per set of augmented SMILES
            smiles_y_pred_mean, _ = utils.mean_median_result(smiles_x_enum_card, smiles_y_pred)

            # unscale prediction's outcomes
            smiles_y_pred_mean = self.scalers_list[ifold].inverse_transform(smiles_y_pred_mean.reshape(-1,1))

            smiles_y_pred_mean_array = np.append(smiles_y_pred_mean_array, smiles_y_pred_mean.reshape(1,-1), axis = 0)

            if ifold == (self.k_fold_number-1):
                smiles_y_pred_mean_ensemble = np.mean(smiles_y_pred_mean_array, axis = 0)
                smiles_y_pred_sd_ensemble = np.std(smiles_y_pred_mean_array, axis = 0)

                pred_from_ens = pd.DataFrame(data=[smiles_x,
                                                   smiles_y_pred_mean_ensemble,
                                                   smiles_y_pred_sd_ensemble]).T
                pred_from_ens.columns = ['SMILES', 'ens_pred_mean', 'ens_pred_sd']

                print("****************************************")
                print("***Inference of SMILES property done.***")
                print("****************************************\n")

                return pred_from_ens
##
