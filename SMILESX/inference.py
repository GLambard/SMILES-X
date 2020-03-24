import numpy as np
import pandas as pd
import os
import glob

from rdkit import Chem

from tensorflow.keras.models import load_model
from tensorflow.keras import metrics

from SMILESX import utils, model, token, augm

from pickle import load

## Inference on the SMILESX predictions
# smiles_list: targeted SMILES list for property inference (Default: ['CC','CCC','C=O'])
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for inference (Default: None, i.e. automatically detect k_fold_number from main.Main phase)
# augmentation: SMILES's augmentation (Default: False)
# indir: directory of already trained prediction models (*.hdf5) and vocabulary (*.txt) (Default: '../data/')
# outdir: directory for outputs (plots + .txt files) -> 'Inference/'+'{}/{}/'.format(data_name,p_dir_temp) is then created (Default: '../data/')
# returns:
#         Array of SMILES with their inferred property (mean, standard deviation) from models ensembling
def Inference(data_name, 
              smiles_list = ['CC','CCC','C=O'], 
              data_units = '',
              k_fold_number = None,
              augmentation = False, 
              indir = "../data/", 
              outdir = "../data/"):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    input_dir = indir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    save_dir = outdir+'Inference/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
    
    for itype in ["txt","hdf5","pkl"]:
        exists_file = glob.glob(input_dir + "*." + itype)
        exists_file_len = len(exists_file)
        if exists_file_len > 0:
            if itype == "hdf5":
                if k_fold_number is None:
                    k_fold_number = exists_file_len
        else:
            print("***Process of inference automatically aborted!***")
            if itype == "hdf5":
                print("The input directory does not contain any trained models (*.hdf5 files).\n")
            else:
                print("The input directory does not contain any vocabulary (*_Vocabulary.txt file) or data scaler (*.pkl file).\n")
            return
    
    print("**************************************")
    print("***SMILES_X for inference starts...***")
    print("**************************************\n")
    
    print("Checking the SMILES list for inference.\n")
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
        with open(save_dir+'rejected_smiles.txt','w') as f:
            for ismiles in smiles_rejected:
                f.write("%s\n" % ismiles)
        print("Check the {} file for {} rejected SMILES.".format(save_dir+'rejected_smiles.txt', len(smiles_rejected)))
                
    if len(smiles_checked) == 0:
        print("***Process of inference automatically aborted!***")
        print("The provided SMILES are all incorrect and could not be sanitized via RDKit.\n")
        return
    
    smiles_x = np.array(smiles_checked)
    smiles_y = np.array([[np.nan]*len(smiles_checked)]).flatten()
     
    # data augmentation or not
    if augmentation == True:
        print("Data augmentation required.")
        canonical = False
        rotation = True
    else:
        print("No data augmentation is required.")
        canonical = True
        rotation = False

    smiles_x_enum, smiles_x_enum_card, smiles_y_enum = \
    augm.Augmentation(smiles_x, smiles_y, canon=canonical, rotate=rotation)

    print("Number of enumerated SMILES: {}.\n".format(smiles_x_enum.shape[0]))
    
    print("Tokenization of SMILES.")
    # Tokenize SMILES 
    smiles_x_enum_tokens = token.get_tokens(smiles_x_enum)

    # Setting up the vocabulary
    # Tokens as a list
    tokens = token.get_vocab(input_dir+data_name+'_Vocabulary.txt')

    # Add 'pad', 'unk' tokens to the existing list
    vocab_size = len(tokens)
    tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)
    print("Full vocabulary: {}, of size: {}.".format(tokens, vocab_size))

    # Transformation of tokenized SMILES to vector of intergers and vice-versa
    token_to_int = token.get_tokentoint(tokens)
    int_to_token = token.get_inttotoken(tokens)
    
    # models ensembling
    smiles_y_pred_mean_array = np.empty(shape=(0,len(smiles_checked)), dtype='float')
    for ifold in range(k_fold_number):
        # Load the scaler
        scaler = load(open('scaler_fold_' + str(ifold) + '.pkl', 'rb'))
        
        # Model's architecture
        model_train = load_model(input_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(ifold)+'.hdf5', 
                                 custom_objects={'AttentionM': model.AttentionM()})

        if ifold == 0:
            # Maximum of length of SMILES to process
            max_length = model_train.layers[0].output_shape[-1][1]
            print("Maximum length of tokenized SMILES: {} tokens.\n".format(max_length))

        model_train.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

        # predict and compare for the training, validation and test sets
        smiles_x_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = smiles_x_enum_tokens, 
                                                             max_length = max_length, 
                                                             vocab = tokens)

        smiles_y_pred = model_train.predict(smiles_x_enum_tokens_tointvec)

        # compute a mean per set of augmented SMILES
        smiles_y_pred_mean, _ = utils.mean_median_result(smiles_x_enum_card, smiles_y_pred)
        
        # unscale prediction's outcomes
        smiles_y_pred_mean = scaler.inverse_transform(smiles_y_pred_mean.reshape(-1,1))
        
        smiles_y_pred_mean_array = np.append(smiles_y_pred_mean_array, smiles_y_pred_mean.reshape(1,-1), axis = 0)
        
        if ifold == (k_fold_number-1):
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
