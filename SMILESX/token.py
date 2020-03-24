import numpy as np
import ast

from SMILESX import utils, augm
from tensorflow.keras import backend as K
import re

## SMILES Tokenizer
# smiles: input SMILES string to tokenize
# returns:
#         list of tokens + terminators in a SMILES
def smiles_tokenizer(smiles):
    # wildcard
    # aliphatic_organic
    # aromatic organic
    # '[' isotope? symbol chiral? hcount? charge? class? ']'
    # bonds
    # ring bonds
    # branch
    # termination character ' '
    patterns = "(\*|" +\
               "N|O|S|P|F|Cl?|Br?|I|" +\
               "b|c|n|o|s|p|" +\
               "\[.*?\]|" +\
               "-|=|#|\$|:|/|\\|\.|" +\
               "[0-9]|\%[0-9]{2}|" +\
               "\(|\))"
    regex = re.compile(patterns)
    tokens = [token for token in regex.findall(smiles)]
    return [' '] + tokens + [' ']

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
# data: provided data (dataframe of: (smiles, property), or (smiles,))
# augmentation: SMILES's augmentation (Default: False)
# token_tofind: targeted token (elements, bonds, etc.) to find in the training set
# verbose: print SMILES containing the targeted token (0: not print or 1: print, default: 1)
# returns:
#         How many SMILES contain the targeted token, and which SMILES if verbose = 1
def TokensFinder(data, 
                 augmentation = False, 
                 token_tofind = '', 
                 verbose = 1):
    
    print("***************************************")
    print("***SMILES_X token's finder starts...***")
    print("***************************************\n") 
    
    # SMILES from data 
    data_smiles = data.smiles.values
    fake_prop = np.ones((data_smiles.shape[0],1)) # for augm.Augmentation function
    
    # data augmentation or not
    if augmentation == True:
        print("Data augmentation required.")
        canonical = False
        rotation = True
    else:
        print("No data augmentation required.")
        canonical = True
        rotation = False

    data_smiles_enum, _, _ = augm.Augmentation(data_smiles, fake_prop, canon=canonical, rotate=rotation)

    print("Tokenization of provided SMILES.\n")
    # Tokenize SMILES per dataset
    data_smiles_enum_tokens = get_tokens(data_smiles_enum)
    
    # Token finder
    print(">>> The finder is processing the search... >>>")
    n_found = 0
    for ismiles in data_smiles_enum_tokens:
        if token_tofind in ismiles:
            n_found += 1
            if verbose == 1: 
                print(''.join(ismiles))
            
    print("\n{} SMILES found with {} token in the training set.\n".format(n_found, token_tofind))
    
    print("**********************************************************")
    print("***SMILES_X token's finder has terminated successfully.***")
    print("**********************************************************\n") 
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