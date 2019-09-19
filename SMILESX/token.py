import numpy as np
import ast

from SMILESX import utils, augm
from keras import backend as K
# import tensorflow as tf

# #from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# K.set_session(sess)  # set this TensorFlow session as the default session for Keras

## SMILES Tokenizer
# smiles: input SMILES string to tokenize
# returns:
#         list of tokens + terminators in a SMILES

# dictionary of tokens from http://opensmiles.org/opensmiles.html (Formal Grammar)
aliphatic_organic = ['B','C','N','O','S','P','F','Cl','Br','I']
aromatic_organic = ['b','c','n','o','s','p']
bracket = ['[',']'] # includes isotope, symbol, chiral, hcount, charge, class
bond = ['-','=','#','$','/','\\','.']
lrb = ['%'] # long ring bonds '%TWODIGITS'
terminator = [' '] # SPACE - start/end of SMILES
wildcard = ['*']
oov = ['oov'] # out-of-vocabulary tokens
#

def smiles_tokenizer(smiles):
    smiles = smiles.replace('\n','') # avoid '\n' if exists in smiles
    # '[...]' as single token
    smiles = smiles.replace(bracket[0],' '+bracket[0]).replace(bracket[1],bracket[1]+' ')
    # '%TWODIGITS' as single token
    lrb_print = [smiles[ic:ic+3] for ic,ichar in enumerate(smiles) if ichar==lrb[0]]
    if len(lrb_print)!=0:
        for ichar in lrb_print:
            smiles = smiles.replace(ichar, ' '+ichar+' ')
    # split SMILES for [...] recognition
    smiles = smiles.split(' ')
    # split fragments other than [...]
    splitted_smiles = list()
    for ifrag in smiles:
        ifrag_tag = False
        for inac in bracket+lrb:
            if inac in ifrag: 
                ifrag_tag = True
                break
        if ifrag_tag == False:
            # check for Cl, Br in alphatic branches to not dissociate letters (e.g. Cl -> C, l is prohibited)
            for iaa in aliphatic_organic[7:9]:
                ifrag = ifrag.replace(iaa, ' '+iaa+' ')
            ifrag_tmp = ifrag.split(' ')
            for iifrag_tmp in ifrag_tmp:
                if iifrag_tmp!=aliphatic_organic[7] \
                and iifrag_tmp!=aliphatic_organic[8]: # not 'Cl' and not 'Br'
                    splitted_smiles.extend(iifrag_tmp) # automatic split char by char
                else:
                    splitted_smiles.extend([iifrag_tmp])
        else:
            splitted_smiles.extend([ifrag]) # keep the original token size
    return terminator+splitted_smiles+terminator # add start + ... + end of SMILES
##

## Get tokens from list of tokens from SMILES
# smiles_array: array of SMILES to split as individual tokens
# split_l: number of tokens present in a split (default: 1), 
# e.g. split_l = 1 -> np.array(['CC=O']) => [[' ', 'C', 'C', '=', 'O', ' ']], 
# split_l = 2 -> np.array(['CC=O']) => [[' C', 'CC', 'C=', '=O', 'O ']], 
# split_l = 3 -> np.array(['CC=O']) => [[' CC', 'CC=', 'C=O', '=O ']], 
# etc.
# returns:
#         List of tokenized SMILES (=^def list of tokens)
def get_tokens(smiles_array, split_l = 1):
    tokenized_smiles_list = list()
    for ismiles in smiles_array.tolist():
        tokenized_smiles_tmp = smiles_tokenizer(ismiles)
        tokenized_smiles_list.append([''.join(tokenized_smiles_tmp[i:i+split_l])
                                  for i in range(0,len(tokenized_smiles_tmp)-split_l+1,1)
                                 ])
    return tokenized_smiles_list
##

## Vocabulary extraction
# lltokens: list of lists of tokens (list of tokenized SMILES)
# returns:
#         set of individual tokens forming a vocabulary
def extract_vocab(lltokens):
    return set([itoken for ismiles in lltokens for itoken in ismiles])
##

## Dictionary from tokens to integers, and the opposite
# tokens: list of tokens
# returns: 
#         dictionary from tokens to integers
def get_tokentoint(tokens):
    return dict((c, i) for i, c in enumerate(tokens))
# returns:
#         dictionary from integers to tokens
def get_inttotoken(tokens): 
    return dict((i, c) for i, c in enumerate(tokens))
##

## Tokens finder
# data: provided data (numpy array of: (SMILES, property))
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for cross-validation
# k_fold_index: k-fold index to be used for visualization
# augmentation: SMILES's augmentation (Default: False)
# token_tofind: targeted token (elements, bonds, etc.) to find in the training set
# verbose: print SMILES containing the targeted token (0: not print or 1: print, default: 1)
# returns:
#         How many SMILES contain the targeted token, and which SMILES if verbose = 1
def TokensFinder(data, 
                 data_name, 
                 data_units = '',
                 k_fold_number = 8,
                 k_fold_index = 0,
                 augmentation = False, 
                 token_tofind = '', 
                 verbose = 1):
    
    print("***SMILES_X token's finder starts...***\n\n")
    np.random.seed(seed=123)
    seed_list = np.random.randint(int(1e6), size = k_fold_number).tolist()
    
    print("******")
    print("***Fold #{} initiated...***".format(k_fold_index))
    print("******")

    print("***Sampling and splitting of the dataset.***\n")
    # Reproducing the data split of the requested fold (k_fold_index)
    x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = \
    utils.random_split(smiles_input=data.smiles, 
                       prop_input=np.array(data.iloc[:,1]), 
                       random_state=seed_list[k_fold_index], 
                       scaling = True)
    
    # data augmentation or not
    if augmentation == True:
        print("***Data augmentation.***\n")
        canonical = False
        rotation = True
    else:
        print("***No data augmentation has been required.***\n")
        canonical = True
        rotation = False

    x_train_enum, x_train_enum_card, y_train_enum = \
    augm.Augmentation(x_train, y_train, canon=canonical, rotate=rotation)

    x_valid_enum, x_valid_enum_card, y_valid_enum = \
    augm.Augmentation(x_valid, y_valid, canon=canonical, rotate=rotation)

    x_test_enum, x_test_enum_card, y_test_enum = \
    augm.Augmentation(x_test, y_test, canon=canonical, rotate=rotation)

    print("Enumerated SMILES:\n\tTraining set: {}\n\tValidation set: {}\n\tTest set: {}\n".\
    format(x_train_enum.shape[0], x_valid_enum.shape[0], x_test_enum.shape[0]))

    print("***Tokenization of SMILES.***\n")
    # Tokenize SMILES per dataset
    x_train_enum_tokens = get_tokens(x_train_enum)
    x_valid_enum_tokens = get_tokens(x_valid_enum)
    x_test_enum_tokens = get_tokens(x_test_enum)

    print("Examples of tokenized SMILES from a training set:\n{}\n".\
    format(x_train_enum_tokens[:5]))

    # Vocabulary size computation
    all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens
    tokens = extract_vocab(all_smiles_tokens)
    vocab_size = len(tokens)

    train_unique_tokens = list(extract_vocab(x_train_enum_tokens))
    
    # Token finder
    print("The finder is processing the search...")
    n_found = 0
    for ismiles in x_train_enum_tokens:
        if token_tofind in ismiles:
            n_found += 1
            if verbose == 1: 
                print(''.join(ismiles))
            
    print("\n{} SMILES found with {} token in the training set.".format(n_found, token_tofind))
##

## Save the vocabulary for further use of a model
# vocab: vocabulary (list of tokens to save)
# tftokens: text file name with directory to be saved (*.txt)
def save_vocab(vocab, tftokens):
    with open(tftokens,'w') as f_toks:
        f_toks.write(str(list(vocab)))
##        
        
## Get the vocabulary previously saved
# tftokens: text file name with directory in which the vocabulary is saved (*.txt)
# returns: 
#         set of individual tokens forming a vocabulary
def get_vocab(tftokens):
    with open(tftokens,'r') as f_toks:
        tokens = ast.literal_eval(f_toks.read())
    return tokens
##

## Add tokens for unknown ('unk') and extra padding ('pad')
# tokens: list of tokens
# vocab_size: vocabulary size before the addition
# returns:
#         extended vocabulary
#         vocabulary size after extension
def add_extra_tokens(tokens, vocab_size):
    tokens.insert(0,'unk')
    tokens.insert(0,'pad')
    vocab_size = vocab_size+2
    return tokens, vocab_size
##


## Encode SMILES as a vector of integers
# tokenized_smiles_list: list of tokenized SMILES
# max_length: force the vectors to have a same length
# vocab: vocabulary of tokens
# returns: 
#         array of integers of dimensions (number_of_SMILES, max_length)
def int_vec_encode(tokenized_smiles_list, max_length, vocab):
    token_to_int = get_tokentoint(vocab)
    int_smiles_array = np.zeros((len(tokenized_smiles_list),max_length), dtype=np.int32)
    for csmiles,ismiles in enumerate(tokenized_smiles_list):
        ismiles_tmp = list()
        if len(ismiles)<= max_length:
            ismiles_tmp = ['pad']*(max_length-len(ismiles))+ismiles # Force output vectors to have same length
        else:
            ismiles_tmp = ismiles[-max_length:] # longer vectors are truncated (to be changed...)
        integer_encoded = [token_to_int[itoken] if(itoken in vocab) \
                           else token_to_int['unk']\
                           for itoken in ismiles_tmp]
        int_smiles_array[csmiles] = integer_encoded
    
    return int_smiles_array