import pandas as pd
import numpy as np

import os
# For fixing the GPU in use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use (e.g. "0", "1", etc.)
os.environ["CUDA_VISIBLE_DEVICES"]="0";  

import math

import collections

import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.random import seed
seed(12345)

import GPy, GPyOpt

from sklearn.metrics import r2_score
np.set_printoptions(precision=3)
from sklearn.preprocessing import RobustScaler

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation

from itertools import cycle
from adjustText import adjust_text

from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import MinMaxScaler

from rdkit import Chem
from rdkit.Chem import Draw

import ast

from scipy.ndimage.interpolation import shift

from keras.utils import Sequence

from keras.models import Model
from keras.layers import Input, Dropout, Dense, Softmax, Multiply, Add, RepeatVector
from keras.layers import Embedding, Lambda
from keras.layers.wrappers import Bidirectional
from keras.layers import CuDNNLSTM, TimeDistributed, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.engine.topology import Layer

import multiprocessing

from keras.utils import np_utils
from keras.utils import multi_gpu_model

from keras import metrics

from keras import backend as K
import tensorflow as tf

from keras.models import load_model

##
#from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras
##

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

## Rotate atoms' index in a list
# li: List to be rotated
# x: Index to be placed first in the list
def rotate_atoms(li, x):
    return (li[x%len(li):]+li[:x%len(li)])
##

## Generate SMILES list
# smiles: SMILES list to be prepared
# kekule: kekulize (default: False)
# canon: canonicalize (default: True)
# rotate: rotation of atoms's index for augmentation (default: False)
# returns:
#         list of augmented SMILES (non-canonical equivalents from canonical SMILES representation)
def generate_smiles(smiles, kekule = False, canon = True, rotate = False):
    smiles_list = list()
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    if mol != None: 
        n_atoms = mol.GetNumAtoms()
        n_atoms_list = [nat for nat in range(n_atoms)]
        if rotate == True:
            canon = False 
            if n_atoms != 0:
                for iatoms in range(n_atoms):
                    n_atoms_list_tmp = rotate_atoms(n_atoms_list,iatoms) # rotate atoms' index
                    nmol = Chem.RenumberAtoms(mol,n_atoms_list_tmp) # renumber atoms in mol
                    try:
                        smiles = Chem.MolToSmiles(nmol,
                                                  isomericSmiles = True, # keep isomerism
                                                  kekuleSmiles = kekule, # kekulize or not
                                                  rootedAtAtom = -1, # default
                                                  canonical = canon, # canonicalize or not
                                                  allBondsExplicit = False, # 
                                                  allHsExplicit = False) #
                    except:
                        smiles = 'None'
                    smiles_list.append(smiles)
            else:
                smiles = 'None'
                smiles_list.append(smiles)
        else:
            try:
                smiles = Chem.MolToSmiles(mol,
                                          isomericSmiles = True, 
                                          kekuleSmiles = kekule, 
                                          rootedAtAtom = -1, 
                                          canonical = canon, 
                                          allBondsExplicit = False, 
                                          allHsExplicit = False)
            except:
                smiles = 'None'
            smiles_list.append(smiles)
    else:
        smiles = 'None'
        smiles_list.append(smiles)
    
    smiles_list = pd.DataFrame(smiles_list).drop_duplicates().iloc[:,0].values.tolist() # duplicates are discarded
    
    return smiles_list
##

## Augmentation
# smiles_array: SMILES array for augmentation
# prop_array: property array for augmentation
# canon: canonicalize (default: True)
# rotate: rotation of atoms' index for augmentation (default: False)
# returns:
#         array of augmented SMILES, 
#         number of augmentation per SMILES, 
#         array of related property
def Augmentation(smiles_array, prop_array, canon = True, rotate = False):
    smiles_enum = list()
    prop_enum = list()
    smiles_enum_card = list()
    for csmiles,ismiles in enumerate(smiles_array.tolist()):
        enumerated_smiles = generate_smiles(ismiles, canon = canon, rotate = rotate)
        if 'None' not in enumerated_smiles:
            smiles_enum_card.append(len(enumerated_smiles))
            smiles_enum.extend(enumerated_smiles)
            prop_enum.extend([prop_array[csmiles]]*len(enumerated_smiles))

    return np.array(smiles_enum), smiles_enum_card, np.array(prop_enum)
##

## Vocabulary extraction
# lltokens: list of lists of tokens (list of tokenized SMILES)
# returns:
#         set of individual tokens forming a vocabulary
def extract_vocab(lltokens):
    return set([itoken for ismiles in lltokens for itoken in ismiles])
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
##

## Data sequence to be fed to the neural network during training through batches of data
class DataSequence(Sequence):
    # Initialization
    # smiles_set: array of tokenized SMILES of dimensions (number_of_SMILES, max_length)
    # vocab: vocabulary of tokens
    # max_length: maximum length for SMILES in the dataset
    # props_set: array of targeted property
    # batch_size: batch's size
    # soft_padding: pad tokenized SMILES at the same length in the whole dataset (False), or in the batch only (True) (Default: False)
    # returns: 
    #         a batch of arrays of tokenized and encoded SMILES, 
    #         a batch of SMILES property
    def __init__(self, smiles_set, vocab, max_length, props_set, batch_size, soft_padding = False):
        self.smiles_set = smiles_set
        self.vocab = vocab
        self.max_length = max_length
        self.props_set = props_set
        self.batch_size = batch_size
        self.iepoch = 0
        self.soft_padding = soft_padding

    def on_epoch_end(self):
        self.iepoch += 1
        
    def __len__(self):
        return int(np.ceil(len(self.smiles_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        tokenized_smiles_list_tmp = self.smiles_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        # self.max_length + 1 padding
        if self.soft_padding:
            max_length_tmp = np.max([len(ismiles) for ismiles in tokenized_smiles_list_tmp])
        else:
            max_length_tmp = self.max_length
        batch_x = int_vec_encode(tokenized_smiles_list = tokenized_smiles_list_tmp, 
                                 max_length = max_length_tmp+1,
                                 vocab = self.vocab)
        #batch_x = self.batch[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.props_set[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)
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

## Custom attention layer
# modified from https://github.com/sujitpal/eeap-examples
class AttentionM(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix.
    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)
    # Usage
        enc = LSTM(EMBED_SIZE, return_sequences=True)(...)
        att = AttentionM()(enc)
    """    
    def __init__(self, return_probabilities = False, **kwargs):
        self.return_probabilities = return_probabilities
        super(AttentionM, self).__init__(**kwargs)

    
    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS,)
        self.W = self.add_weight(name="W_{:s}".format(self.name), 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionM, self).build(input_shape)


    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, 1)
        atx = K.expand_dims(at, axis=-1)
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ot = x * atx
        # output: (BATCH_SIZE, EMBED_SIZE)
        if self.return_probabilities: 
            return atx # for visualization of the attention weights
        else:
            return K.sum(ot, axis=1) # for prediction

    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


    def get_config(self):
        return super(AttentionM, self).get_config()
##
    
## Neural architecture of the SMILES-X
class LSTMAttModel():
    # Initialization
    # inputtokens: maximum length for the encoded and tokenized SMILES
    # vocabsize: size of the vocabulary
    # lstmunits: number of LSTM units
    # denseunits: number of dense units
    # embedding: dimension of the embedded vectors
    # return_proba: return the attention vector (True) or not (False) (Default: False)
    # Returns: 
    #         a model in the Keras API format
    @staticmethod
    def create(inputtokens, vocabsize, lstmunits=16, denseunits=16, embedding=32, return_proba = False):

        input_ = Input(shape=(inputtokens,), dtype='int32')

        # Embedding layer
        net = Embedding(input_dim=vocabsize, 
                        output_dim=embedding, 
                        input_length=inputtokens)(input_)

        # Bidirectional LSTM layer
        net = Bidirectional(CuDNNLSTM(lstmunits, return_sequences=True))(net)
        net = TimeDistributed(Dense(denseunits))(net)
        net = AttentionM(return_probabilities=return_proba)(net)

        # Output layer
        net = Dense(1, activation="linear")(net)
        model = Model(inputs=input_, outputs=net)

        return model
##

## Function to fit a model on a multi-GPU machine
class ModelMGPU(Model):
    # Initialization
    # ser_model: based model to pass to >1 GPUs
    # gpus: number of GPUs
    # bridge_type: optimize for bridge types (NVLink or not) 
    # returns:
    #         a multi-GPU model (based model copied to GPUs, batch is splitted over the GPUs)
    def __init__(self, ser_model, gpus, bridge_type):
        if bridge_type == 'NVLink':
            pmodel = multi_gpu_model(ser_model, gpus, cpu_merge=False) # recommended for NV-link
        else:
            pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
##
    
## Attention weights depiction
# from https://github.com/rdkit/rdkit/blob/24f1737839c9302489cadc473d8d9196ad9187b4/rdkit/Chem/Draw/SimilarityMaps.py
# returns:
#         a similarity map for a molecule given the attention weights
def GetSimilarityMapFromWeights(mol, weights, colorMap=None, scale=-1, size=(250, 250),
                                sigma=None, coordScale=1.5, step=0.01, colors='k', contourLines=10,
                                alpha=0.5, **kwargs):
    """
    Generates the similarity map for a molecule given the atomic weights.
    Parameters:
    mol -- the molecule of interest
    colorMap -- the matplotlib color map scheme, default is custom PiWG color map
    scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                          scale = double -> this is the maximum scale
    size -- the size of the figure
    sigma -- the sigma for the Gaussians
    coordScale -- scaling factor for the coordinates
    step -- the step for calcAtomGaussian
    colors -- color of the contour lines
    contourLines -- if integer number N: N contour lines are drawn
                    if list(numbers): contour lines at these numbers are drawn
    alpha -- the alpha blending value for the contour lines
    kwargs -- additional arguments for drawing
    """
    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")
    fig = Draw.MolToMPL(mol, coordScale=coordScale, size=size, **kwargs)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = 0.3 * math.sqrt(
                    sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i])**2 for i in range(2)]))
        else:
            sigma = 0.3 * math.sqrt(sum([(mol._atomPs[0][i] - mol._atomPs[1][i])**2 for i in range(2)]))
        sigma = round(sigma, 2)
    x, y, z = Draw.calcAtomGaussians(mol, sigma, weights=weights, step=step)
    # scaling
    if scale <= 0.0:
        maxScale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
        minScale = min(math.fabs(np.min(z)), math.fabs(np.max(z)))
    else:
        maxScale = scale
    
    fig.axes[0].imshow(z, cmap=colorMap, interpolation='bilinear', origin='lower',
                     extent=(0, 1, 0, 1), vmin=minScale, vmax=maxScale)
    # contour lines
    # only draw them when at least one weight is not zero
    if len([w for w in weights if w != 0.0]):
        contourset = fig.axes[0].contour(x, y, z, contourLines, colors=colors, alpha=alpha, **kwargs)
        for j, c in enumerate(contourset.collections):
            if contourset.levels[j] == 0.0:
                c.set_linewidth(0.0)
            elif contourset.levels[j] < 0:
                c.set_dashes([(0, (3.0, 3.0))])
    fig.axes[0].set_axis_off()
    return fig
##

## Visualization of the Embedding layer 
# data: provided data (numpy array of: (SMILES, property))
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for cross-validation
# k_fold_index: k-fold index to be used for visualization
# augmentation: SMILES's augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Embedding_Vis/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# affinity_propn: Affinity propagation tagging (Default: True)
# returns:
#         PCA visualization of a representation of SMILES tokens from the embedding layer
def Embedding_Vis(data, 
                  data_name, 
                  data_units = '',
                  k_fold_number = 8,
                  k_fold_index = 0,
                  augmentation = False, 
                  outdir = "../data/", 
                  affinity_propn = True, 
                  verbose = 0):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    input_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    save_dir = outdir+'Embedding_Vis/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
    
    print("***SMILES_X for embedding visualization starts...***\n\n")
    np.random.seed(seed=123)
    seed_list = np.random.randint(int(1e6), size = k_fold_number).tolist()
    # Train/validation/test data splitting - 80/10/10 % at random with diff. seeds for k_fold_number times
    selection_seed = seed_list[k_fold_index]
        
    print("******")
    print("***Fold #{} initiated...***".format(selection_seed))
    print("******")

    print("***Sampling and splitting of the dataset.***\n")
    x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = \
    random_split(smiles_input=data.smiles, 
                 prop_input=np.array(data.iloc[:,1]), 
                 random_state=selection_seed, 
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
    Augmentation(x_train, y_train, canon=canonical, rotate=rotation)

    x_valid_enum, x_valid_enum_card, y_valid_enum = \
    Augmentation(x_valid, y_valid, canon=canonical, rotate=rotation)

    x_test_enum, x_test_enum_card, y_test_enum = \
    Augmentation(x_test, y_test, canon=canonical, rotate=rotation)

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
    print(train_unique_tokens)
    print("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
    train_unique_tokens.insert(0,'pad')
    
    # Tokens as a list
    tokens = get_vocab(input_dir+data_name+'_tokens_set_seed'+str(selection_seed)+'.txt')
    # Add 'pad', 'unk' tokens to the existing list
    tokens, vocab_size = add_extra_tokens(tokens, vocab_size)
    
    print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))

    # Maximum of length of SMILES to process
    max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
    print("Maximum length of tokenized SMILES: {} tokens (termination spaces included)\n".format(max_length))

    # Transformation of tokenized SMILES to vector of intergers and vice-versa
    token_to_int = get_tokentoint(tokens)
    int_to_token = get_inttotoken(tokens)

    model = load_model(input_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(selection_seed)+'.hdf5', 
                       custom_objects={'AttentionM': AttentionM()})

    print("Chosen model summary:\n")
    print(model.summary())
    print("\n")

    print("***Embedding of the individual tokens from the chosen model.***\n")
    model.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

    model_embed_weights = model.layers[1].get_weights()[0]
    #print(model_embed_weights.shape)
    #tsne = TSNE(perplexity=30, early_exaggeration=120 , n_components=2, random_state=123, verbose=0)
    pca = PCA(n_components=2, random_state=123)
    transformed_weights = pca.fit_transform(model_embed_weights)
    #transformed_weights = tsne.fit_transform(model_embed_weights)    
    
    f = plt.figure(figsize=(9, 9))
    ax = plt.subplot(aspect='equal')
    
    if affinity_propn:
        # Compute Affinity Propagation
        af = AffinityPropagation().fit(model_embed_weights)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters_ = len(cluster_centers_indices)
        # Plot it
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            class_members = np.where(np.array(labels == k) == True)[0].tolist()
            for ilabpt in class_members:
                alpha_tmp = 0.5 if tokens[ilabpt] in train_unique_tokens else 0.5
                line_tmp = 1 if tokens[ilabpt] in train_unique_tokens else 5
                marker_tmp = 'o' if tokens[ilabpt] in train_unique_tokens else 'x'
                edge_color_tmp = 'black' if tokens[ilabpt] in train_unique_tokens else col
                ax.plot(transformed_weights[ilabpt, 0], 
                        transformed_weights[ilabpt, 1], col, 
                        marker=marker_tmp, markeredgecolor = edge_color_tmp, markeredgewidth=line_tmp, 
                        alpha=alpha_tmp, markersize=10)
    else:
        # Black and white plot
        for ilabpt in range(vocab_size):
            alpha_tmp = 0.5 if tokens[ilabpt] in train_unique_tokens else 0.2
            size_tmp = 40 if tokens[ilabpt] in train_unique_tokens else 20
            ax.scatter(transformed_weights[ilabpt,0], transformed_weights[ilabpt,1], 
                       lw=1, s=size_tmp, facecolor='black', marker='o', alpha=alpha_tmp)
    
    annotations = []
    weight_tmp = 'bold'
    ilabpt = 0
    for ilabpt, (x_i, y_i) in enumerate(zip(transformed_weights[:,0].tolist(), 
                                            transformed_weights[:,1].tolist())):
        weight_tmp = 'black' if tokens[ilabpt] in train_unique_tokens else 'normal'
        tokens_tmp = tokens[ilabpt]
        if tokens_tmp == ' ':
            tokens_tmp = 'space'
        elif tokens_tmp == '.':
            tokens_tmp = 'dot'
        annotations.append(plt.text(x_i,y_i, tokens_tmp, fontsize=12, weight=weight_tmp))
    adjust_text(annotations,
                x=transformed_weights[:,0].tolist(),y=transformed_weights[:,1].tolist(), 
                arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    
    plt.xticks([])
    plt.yticks([])
    ax.axis('tight')
    
    plt.savefig(save_dir+'Visualization_'+data_name+'_Embedding_seed_'+str(selection_seed)+'.png', bbox_inches='tight')
    plt.show()
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
                 k_fold_index=0,
                 augmentation = False, 
                 token_tofind = '', 
                 verbose = 1):
    
    print("***SMILES_X token's finder starts...***\n\n")
    np.random.seed(seed=123)
    seed_list = np.random.randint(int(1e6), size = k_fold_number).tolist()
    # Train/validation/test data splitting - 80/10/10 % at random with diff. seeds for k_fold_number times
    selection_seed = seed_list[k_fold_index]
        
    print("******")
    print("***Fold #{} initiated...***".format(selection_seed))
    print("******")

    print("***Sampling and splitting of the dataset.***\n")
    x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = \
    random_split(smiles_input=data.smiles, 
                 prop_input=np.array(data.iloc[:,1]), 
                 random_state=selection_seed, 
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
    Augmentation(x_train, y_train, canon=canonical, rotate=rotation)

    x_valid_enum, x_valid_enum_card, y_valid_enum = \
    Augmentation(x_valid, y_valid, canon=canonical, rotate=rotation)

    x_test_enum, x_test_enum_card, y_test_enum = \
    Augmentation(x_test, y_test, canon=canonical, rotate=rotation)

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

## Interpretation of the SMILESX predictions
# data: provided data (numpy array of: (SMILES, property))
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for cross-validation
# k_fold_index: k-fold index to be used for visualization
# augmentation: SMILES's augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Interpretation/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# smiles_toviz: targeted SMILES to visualize (Default: 'CCC')
# font_size: font's size for writing SMILES tokens (Default: 15)
# font_rotation: font's orientation (Default: 'horizontal')
# returns:
#         The 1D and 2D attention maps 
#             The redder and darker the color is, 
#             the stronger is the attention on a given token. 
#         The temporal relative distance Tdist 
#             The closer to zero is the distance value, 
#             the closer is the temporary prediction on the SMILES fragment to the whole SMILES prediction.
def Interpretation(data, 
                   data_name, 
                   data_units = '',
                   k_fold_number = 8,
                   k_fold_index=0,
                   augmentation = False, 
                   outdir = "../data/", 
                   smiles_toviz = 'CCC', 
                   font_size = 15, 
                   font_rotation = 'horizontal'):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    input_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    save_dir = outdir+'Interpretation/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
    
    print("***SMILES_X Interpreter starts...***\n\n")
    np.random.seed(seed=123)
    seed_list = np.random.randint(int(1e6), size = k_fold_number).tolist()
    # Train/validation/test data splitting - 80/10/10 % at random with diff. seeds for k_fold_number times
    selection_seed = seed_list[k_fold_index]
        
    print("******")
    print("***Fold #{} initiated...***".format(selection_seed))
    print("******")

    print("***Sampling and splitting of the dataset.***\n")
    x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = \
    random_split(smiles_input=data.smiles, 
                 prop_input=np.array(data.iloc[:,1]), 
                 random_state=selection_seed, 
                 scaling = True)

    np.savetxt(save_dir+'smiles_train.txt', np.asarray(x_train), newline="\n", fmt='%s')
    np.savetxt(save_dir+'smiles_valid.txt', np.asarray(x_valid), newline="\n", fmt='%s')
    np.savetxt(save_dir+'smiles_test.txt', np.asarray(x_test), newline="\n", fmt='%s')
    
    mol_toviz = Chem.MolFromSmiles(smiles_toviz)
    if mol_toviz != None:
        smiles_toviz_can = Chem.MolToSmiles(mol_toviz)
    else:
        print("***Process of visualization automatically aborted!***")
        print("The smiles_toviz is incorrect and cannot be canonicalized by RDKit.")
        return
    smiles_toviz_x = np.array([smiles_toviz_can])
    if smiles_toviz_can in np.array(data.smiles):
        smiles_toviz_y = np.array([[data.iloc[np.where(data.smiles == smiles_toviz_x[0])[0][0],1]]])
    else:
        smiles_toviz_y = np.array([[np.nan]])

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
    Augmentation(x_train, y_train, canon=canonical, rotate=rotation)

    x_valid_enum, x_valid_enum_card, y_valid_enum = \
    Augmentation(x_valid, y_valid, canon=canonical, rotate=rotation)

    x_test_enum, x_test_enum_card, y_test_enum = \
    Augmentation(x_test, y_test, canon=canonical, rotate=rotation)
    
    smiles_toviz_x_enum, smiles_toviz_x_enum_card, smiles_toviz_y_enum = \
    Augmentation(smiles_toviz_x, smiles_toviz_y, canon=canonical, rotate=rotation)

    print("Enumerated SMILES:\n\tTraining set: {}\n\tValidation set: {}\n\tTest set: {}\n".\
    format(x_train_enum.shape[0], x_valid_enum.shape[0], x_test_enum.shape[0]))

    print("***Tokenization of SMILES.***\n")
    # Tokenize SMILES per dataset
    x_train_enum_tokens = get_tokens(x_train_enum)
    x_valid_enum_tokens = get_tokens(x_valid_enum)
    x_test_enum_tokens = get_tokens(x_test_enum)
    
    smiles_toviz_x_enum_tokens = get_tokens(smiles_toviz_x_enum)

    print("Examples of tokenized SMILES from a training set:\n{}\n".\
    format(x_train_enum_tokens[:5]))

    # Vocabulary size computation
    all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens
    tokens = extract_vocab(all_smiles_tokens)
    vocab_size = len(tokens)

    train_unique_tokens = list(extract_vocab(x_train_enum_tokens))
    print(train_unique_tokens)
    print("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
    train_unique_tokens.insert(0,'pad')
    
    # Tokens as a list
    tokens = get_vocab(input_dir+data_name+'_tokens_set_seed'+str(selection_seed)+'.txt')
    # Add 'pad', 'unk' tokens to the existing list
    tokens, vocab_size = add_extra_tokens(tokens, vocab_size)
    
    print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))

    # Maximum of length of SMILES to process
    max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
    print("Maximum length of tokenized SMILES: {} tokens\n".format(max_length))

    # Transformation of tokenized SMILES to vector of intergers and vice-versa
    token_to_int = get_tokentoint(tokens)
    int_to_token = get_inttotoken(tokens)

    # Best architecture to visualize from
    model_topredict = load_model(input_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(selection_seed)+'.hdf5', 
                           custom_objects={'AttentionM': AttentionM()})
    best_arch = [model_topredict.layers[2].output_shape[-1]/2, 
                 model_topredict.layers[3].output_shape[-1], 
                 model_topredict.layers[1].output_shape[-1]]

    # Architecture to return attention weights
    model = LSTMAttModel.create(inputtokens = max_length+1, 
                                vocabsize = vocab_size, 
                                lstmunits= int(best_arch[0]), 
                                denseunits = int(best_arch[1]), 
                                embedding = int(best_arch[2]), 
                                return_proba = True)

    print("Best model summary:\n")
    print(model.summary())
    print("\n")

    print("***Interpretation from the best model.***\n")
    model.load_weights(input_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(selection_seed)+'.hdf5')
    model.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

    smiles_toviz_x_enum_tokens_tointvec = int_vec_encode(tokenized_smiles_list= smiles_toviz_x_enum_tokens, 
                                                         max_length = max_length+1,
                                                         vocab = tokens)
    
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[-2].output)
    intermediate_output = intermediate_layer_model.predict(smiles_toviz_x_enum_tokens_tointvec)
    
    smiles_toviz_x_card_cumsum_viz = np.cumsum(smiles_toviz_x_enum_card)
    smiles_toviz_x_card_cumsum_shift_viz = shift(smiles_toviz_x_card_cumsum_viz, 1, cval=0)

    mols_id = 0
    ienumcard = smiles_toviz_x_card_cumsum_shift_viz[mols_id]
    
    smiles_len_tmp = len(smiles_toviz_x_enum_tokens[ienumcard])
    intermediate_output_tmp = intermediate_output[ienumcard,-smiles_len_tmp+1:-1].flatten().reshape(1,-1)
    max_intermediate_output_tmp = np.max(intermediate_output_tmp)

    plt.matshow(intermediate_output_tmp, 
                cmap='Reds')
    plt.tick_params(axis='x', bottom = False)
    plt.xticks([ix for ix in range(smiles_len_tmp-2)])
    plt.xticks(range(smiles_len_tmp-2), 
               [int_to_token[iint].replace('pad','') \
                for iint in smiles_toviz_x_enum_tokens_tointvec[ienumcard,-smiles_len_tmp+1:-1]], 
               fontsize = font_size, 
               rotation = font_rotation)
    plt.yticks([])
    plt.savefig(save_dir+'Interpretation_1D_'+data_name+'_seed_'+str(selection_seed)+'.png', bbox_inches='tight')
    #plt.show()
    
    smiles_tmp = smiles_toviz_x_enum[ienumcard]
    mol_tmp = Chem.MolFromSmiles(smiles_tmp)
    smiles_len_tmp = len(smiles_toviz_x_enum_tokens[ienumcard])
    mol_df_tmp = pd.DataFrame([smiles_toviz_x_enum_tokens[ienumcard][1:-1],intermediate_output[ienumcard].\
                               flatten().\
                               tolist()[-smiles_len_tmp+1:-1]]).transpose()
    bond = ['-','=','#','$','/','\\','.','(',')']
    mol_df_tmp = mol_df_tmp[~mol_df_tmp.iloc[:,0].isin(bond)]
    mol_df_tmp = mol_df_tmp[[not itoken.isdigit() for itoken in mol_df_tmp.iloc[:,0].values.tolist()]]

    minmaxscaler = MinMaxScaler(feature_range=(0,1))
    norm_weights = minmaxscaler.fit_transform(mol_df_tmp.iloc[:,1].values.reshape(-1,1)).flatten().tolist()
    fig = GetSimilarityMapFromWeights(mol=mol_tmp, 
                                      size = (250,250), 
                                      scale=-1,  
                                      sigma=0.05,
                                      weights=norm_weights, 
                                      colorMap='Reds', 
                                      contourLines = 10,
                                      alpha = 0.25)
    fig.savefig(save_dir+'Interpretation_2D_'+data_name+'_seed_'+str(selection_seed)+'.png', bbox_inches='tight')
    #fig.show()
    
    model_topredict.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])
    
    y_pred_test_tmp = model_topredict.predict(smiles_toviz_x_enum_tokens_tointvec[ienumcard].reshape(1,-1))[0,0]
    y_test_tmp = smiles_toviz_y_enum[ienumcard,0]
    if not np.isnan(y_test_tmp):
        print("True value: {0:.2f} Predicted: {1:.2f}".format(y_test_tmp,
                                                    scaler.inverse_transform(y_pred_test_tmp.reshape(1, -1))[0][0]))
    else:
        print("Predicted: {0:.2f}".format(scaler.inverse_transform(y_pred_test_tmp.reshape(1, -1))[0][0]))
    
    smiles_len_tmp = len(smiles_toviz_x_enum_tokens[ienumcard])
    diff_topred_list = list()
    diff_totrue_list = list()
    for csubsmiles in range(1,smiles_len_tmp):
        isubsmiles = smiles_toviz_x_enum_tokens[ienumcard][:csubsmiles]+[' ']
        isubsmiles_tointvec= int_vec_encode(tokenized_smiles_list = [isubsmiles], 
                                            max_length = max_length+1, 
                                            vocab = tokens)
        predict_prop_tmp = model_topredict.predict(isubsmiles_tointvec)[0,0]
        diff_topred_tmp = (predict_prop_tmp-y_pred_test_tmp)/np.abs(y_pred_test_tmp)
        diff_topred_list.append(diff_topred_tmp)
        diff_totrue_tmp = (predict_prop_tmp-y_test_tmp)/np.abs(y_test_tmp)
        diff_totrue_list.append(diff_totrue_tmp)
    max_diff_topred_tmp = np.max(diff_topred_list)
    max_diff_totrue_tmp = np.max(diff_totrue_list)

    plt.figure(figsize=(15,7))
    markers, stemlines, baseline = plt.stem([ix for ix in range(smiles_len_tmp-1)], 
                                            diff_topred_list, 
                                            'k.-', 
                                             use_line_collection=True)
    plt.setp(baseline, color='k', linewidth=2, linestyle='--')
    plt.setp(markers, linewidth=1, marker='o', markersize=10, markeredgecolor = 'black')
    plt.setp(stemlines, color = 'k', linewidth=0.5, linestyle='-')
    plt.xticks(range(smiles_len_tmp-1), 
               smiles_toviz_x_enum_tokens[ienumcard][:-1],
               fontsize = font_size, 
               rotation = font_rotation)
    plt.yticks(fontsize = 20)
    plt.ylabel('Temporal relative distance', fontsize = 25, labelpad = 15)
    plt.savefig(save_dir+'Interpretation_temporal_'+data_name+'_seed_'+str(selection_seed)+'.png', bbox_inches='tight')
    #plt.show()
##

## Interpretation of the SMILESX predictions
# smiles_list: targeted SMILES list for property inference (Default: ['CC','CCC','C=O'])
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for inference
# augmentation: SMILES's augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Inference/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# returns:
#         Array of SMILES with their inferred property (mean, standard deviation) from models ensembling
def Inference(data_name, 
              smiles_list = ['CC','CCC','C=O'], 
              data_units = '',
              k_fold_number = 8,
              augmentation = False, 
              outdir = "../data/"):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    input_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    save_dir = outdir+'Inference/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
    
    print("***SMILES_X for inference starts...***\n\n")
    np.random.seed(seed=123)
    seed_list = np.random.randint(int(1e6), size = k_fold_number).tolist()
        
    print("***Checking the SMILES list for inference***\n")
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
                
    if len(smiles_checked) == 0:
        print("***Process of inference automatically aborted!***")
        print("The provided SMILES are all incorrect and could not be verified via RDKit.")
        return
    
    smiles_x = np.array(smiles_checked)
    smiles_y = np.array([[np.nan]*len(smiles_checked)]).flatten()
     
    # data augmentation or not
    if augmentation == True:
        print("***Data augmentation.***\n")
        canonical = False
        rotation = True
    else:
        print("***No data augmentation has been required.***\n")
        canonical = True
        rotation = False

    smiles_x_enum, smiles_x_enum_card, smiles_y_enum = \
    Augmentation(smiles_x, smiles_y, canon=canonical, rotate=rotation)

    print("Enumerated SMILES: {}\n".format(smiles_x_enum.shape[0]))
    
    print("***Tokenization of SMILES.***\n")
    # Tokenize SMILES 
    smiles_x_enum_tokens = get_tokens(smiles_x_enum)

    # models ensembling
    smiles_y_pred_mean_array = np.empty(shape=(0,len(smiles_checked)), dtype='float')
    for ifold in range(k_fold_number):
        
        # Tokens as a list
        tokens = get_vocab(input_dir+data_name+'_tokens_set_seed'+str(seed_list[ifold])+'.txt')
        # Add 'pad', 'unk' tokens to the existing list
        vocab_size = len(tokens)
        tokens, vocab_size = add_extra_tokens(tokens, vocab_size)

        # Transformation of tokenized SMILES to vector of intergers and vice-versa
        token_to_int = get_tokentoint(tokens)
        int_to_token = get_inttotoken(tokens)
        
        # Best architecture to visualize from
        model = load_model(input_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(seed_list[ifold])+'.hdf5', 
                               custom_objects={'AttentionM': AttentionM()})

        if ifold == 0:
            # Maximum of length of SMILES to process
            max_length = model.layers[0].output_shape[-1]
            print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))
            print("Maximum length of tokenized SMILES: {} tokens\n".format(max_length))

        model.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

        # predict and compare for the training, validation and test sets
        smiles_x_enum_tokens_tointvec = int_vec_encode(tokenized_smiles_list = smiles_x_enum_tokens, 
                                                      max_length = max_length, 
                                                      vocab = tokens)

        smiles_y_pred = model.predict(smiles_x_enum_tokens_tointvec)

        # compute a mean per set of augmented SMILES
        smiles_y_pred_mean, _ = mean_median_result(smiles_x_enum_card, smiles_y_pred)
        
        smiles_y_pred_mean_array = np.append(smiles_y_pred_mean_array, smiles_y_pred_mean.reshape(1,-1), axis = 0)
        
        if ifold == (k_fold_number-1):
            smiles_y_pred_mean_ensemble = np.mean(smiles_y_pred_mean_array, axis = 0)
            smiles_y_pred_sd_ensemble = np.std(smiles_y_pred_mean_array, axis = 0)

            pred_from_ens = pd.DataFrame(data=[smiles_x,
                                               smiles_y_pred_mean_ensemble,
                                               smiles_y_pred_sd_ensemble]).T
            pred_from_ens.columns = ['SMILES', 'ens_pred_mean', 'ens_pred_sd']
            
            print("***Inference of SMILES property done.***")
            
            return pred_from_ens
##

## SMILESX main pipeline
# data: provided data (numpy array of: (SMILES, property))
# data_name: dataset's name
# bayopt_bounds: bounds contraining the Bayesian search of neural architectures
# data_units: property's SI units
# k_fold_number: number of k-folds used for cross-validation (Default: 8)
# augmentation: SMILES augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Main/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# bayopt_n_epochs: number of epochs for training a neural architecture during Bayesian architecture search (Default: 10)
# bayopt_n_rounds: number of architectures to be sampled during Bayesian architecture search (initialization + optimization) (Default: 25)
# bayopt_it_factor: portion of data to be used during Bayesian architecture search (Default: 1)
# bayopt_on: Use Bayesian architecture search or not (Default: True)
# lstmunits_ref: number of LSTM units for the k_fold_index if Bayesian architecture search is off
# denseunits_ref: number of dense units for the k_fold_index if Bayesian architecture search is off
# embedding_ref: number of embedding dimensions for the k_fold_index if Bayesian architecture search is off
# batch_size_ref: batch size for neural architecture training if Bayesian architecture search is off
# alpha_ref: 10^(-alpha_ref) Adam's learning rate for neural architecture training if Bayesian architecture search is off
# n_gpus: number of GPUs to be used in parallel (Default: 1)
# bridge_type: bridge's type to be used by GPUs (e.g. 'NVLink' or 'None') (Default: 'None')
# patience: number of epochs to respect before stopping a training after minimal validation's error (Default: 25)
# n_epochs: maximum of epochs for training (Default: 1000)
# returns:
#         Tokens list (Vocabulary) -> *.txt
#         Best architecture -> *.hdf5
#         Training plot (Loss VS Epoch) -> History_*.png
#         Predictions VS Observations plot -> TrainValidTest_*.png
#
#         per k_fold (e.g. if k_fold_number = 8, 8 versions of these outputs are returned) in outdir
##
def Main(data, 
         data_name, 
         bayopt_bounds, 
         data_units = '',
         k_fold_number = 8, 
         augmentation = False, 
         outdir = "../data/", 
         bayopt_n_epochs = 10,
         bayopt_n_rounds = 25, 
         bayopt_it_factor = 1, 
         bayopt_on = True, 
         lstmunits_ref = 512, 
         denseunits_ref = 512, 
         embedding_ref = 512, 
         batch_size_ref = 64, 
         alpha_ref = 3, 
         n_gpus = 1, 
         bridge_type = 'None', 
         patience = 25, 
         n_epochs = 1000):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    save_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
        
    print("***SMILES_X starts...***\n\n")
    np.random.seed(seed=123)
    seed_list = np.random.randint(int(1e6), size = k_fold_number).tolist()
    # Train/validation/test data splitting - 80/10/10 % at random with diff. seeds for k_fold_number times
    for ifold in range(k_fold_number):
        
        print("******")
        print("***Fold #{} initiated...***".format(ifold))
        print("******")
        
        print("***Sampling and splitting of the dataset.***\n")
        selection_seed = seed_list[ifold]
        x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = \
        random_split(smiles_input=data.smiles, 
                     prop_input=np.array(data.iloc[:,1]), 
                     random_state=selection_seed, 
                     scaling = True)
              
        # data augmentation or not
        if augmentation == True:
            print("***Data augmentation to {}***\n".format(augmentation))
            canonical = False
            rotation = True
        else:
            print("***No data augmentation has been required.***\n")
            canonical = True
            rotation = False
            
        x_train_enum, x_train_enum_card, y_train_enum = \
        Augmentation(x_train, y_train, canon=canonical, rotate=rotation)

        x_valid_enum, x_valid_enum_card, y_valid_enum = \
        Augmentation(x_valid, y_valid, canon=canonical, rotate=rotation)

        x_test_enum, x_test_enum_card, y_test_enum = \
        Augmentation(x_test, y_test, canon=canonical, rotate=rotation)
        
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
        
        train_unique_tokens = extract_vocab(x_train_enum_tokens)
        print("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
        valid_unique_tokens = extract_vocab(x_valid_enum_tokens)
        print("Number of tokens only present in a validation set: {}".format(len(valid_unique_tokens)))
        print("Is the validation set a subset of the training set: {}".\
              format(valid_unique_tokens.issubset(train_unique_tokens)))
        print("What are the tokens by which they differ: {}\n".\
              format(valid_unique_tokens.difference(train_unique_tokens)))
        test_unique_tokens = extract_vocab(x_test_enum_tokens)
        print("Number of tokens only present in a test set: {}".format(len(test_unique_tokens)))
        print("Is the test set a subset of the training set: {}".\
              format(test_unique_tokens.issubset(train_unique_tokens)))
        print("What are the tokens by which they differ: {}".\
              format(test_unique_tokens.difference(train_unique_tokens)))
        print("Is the test set a subset of the validation set: {}".\
              format(test_unique_tokens.issubset(valid_unique_tokens)))
        print("What are the tokens by which they differ: {}\n".\
              format(test_unique_tokens.difference(valid_unique_tokens)))
        
        print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))
        
        # Save the vocabulary for re-use
        save_vocab(tokens, save_dir+data_name+'_tokens_set_seed'+str(selection_seed)+'.txt')
        # Tokens as a list
        tokens = get_vocab(save_dir+data_name+'_tokens_set_seed'+str(selection_seed)+'.txt')
        # Add 'pad', 'unk' tokens to the existing list
        tokens, vocab_size = add_extra_tokens(tokens, vocab_size)
        
        # Maximum of length of SMILES to process
        max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
        print("Maximum length of tokenized SMILES: {} tokens (termination spaces included)\n".format(max_length))
        
        print("***Bayesian Optimization of the SMILESX's architecture.***\n")
        # Transformation of tokenized SMILES to vector of intergers and vice-versa
        token_to_int = get_tokentoint(tokens)
        int_to_token = get_inttotoken(tokens)
        
        if bayopt_on:
            # Operate the bayesian optimization of the neural architecture
            def create_mod(params):
                print('Model: {}'.format(params))

                model_tag = data_name

                K.clear_session()

                if n_gpus > 1:
                    if bridge_type == 'NVLink':
                        model = LSTMAttModel.create(inputtokens = max_length+1, 
                                                    vocabsize = vocab_size, 
                                                    lstmunits=int(params[:,0][0]), 
                                                    denseunits = int(params[:,1]), 
                                                    embedding = int(params[:,2][0]))
                    else:
                        with tf.device('/cpu'): # necessary to multi-GPU scaling
                            model = LSTMAttModel.create(inputtokens = max_length+1, 
                                                        vocabsize = vocab_size, 
                                                        lstmunits=int(params[:,0][0]), 
                                                        denseunits = int(params[:,1]), 
                                                        embedding = int(params[:,2][0]))
                            
                    multi_model = ModelMGPU(model, gpus=n_gpus, bridge_type=bridge_type)
                else: # single GPU
                    model = LSTMAttModel.create(inputtokens = max_length+1, 
                                                vocabsize = vocab_size, 
                                                lstmunits=int(params[:,0][0]), 
                                                denseunits = int(params[:,1]), 
                                                embedding = int(params[:,2][0]))
                    
                    multi_model = model

                batch_size = int(params[:,3][0])
                custom_adam = Adam(lr=math.pow(10,-float(params[:,4][0])))
                multi_model.compile(loss='mse', optimizer=custom_adam, metrics=[metrics.mae,metrics.mse])

                history = multi_model.fit_generator(generator = DataSequence(x_train_enum_tokens,
                                                                             vocab = tokens, 
                                                                             max_length = max_length, 
                                                                             props_set = y_train_enum, 
                                                                             batch_size = batch_size), 
                                                                             steps_per_epoch = math.ceil(len(x_train_enum_tokens)/batch_size)//bayopt_it_factor, 
                                                    validation_data = DataSequence(x_valid_enum_tokens,
                                                                                   vocab = tokens, 
                                                                                   max_length = max_length, 
                                                                                   props_set = y_valid_enum, 
                                                                                   batch_size = min(len(x_valid_enum_tokens), batch_size)),
                                                    validation_steps = math.ceil(len(x_valid_enum_tokens)/min(len(x_valid_enum_tokens), batch_size))//bayopt_it_factor, 
                                                    epochs = bayopt_n_epochs, 
                                                    shuffle = True,
                                                    initial_epoch = 0, 
                                                    verbose = 0)

                best_epoch = np.argmin(history.history['val_loss'])
                mae_valid = history.history['val_mean_absolute_error'][best_epoch]
                mse_valid = history.history['val_mean_squared_error'][best_epoch]
                if math.isnan(mse_valid): # discard diverging architectures (rare event)
                    mae_valid = math.inf
                    mse_valid = math.inf
                print('Valid MAE: {0:0.4f}, RMSE: {1:0.4f}'.format(mae_valid, mse_valid))

                return mse_valid

            print("Random initialization:\n")
            Bayes_opt = GPyOpt.methods.BayesianOptimization(f=create_mod, 
                                                            domain=bayopt_bounds, 
                                                            acquisition_type = 'EI',
                                                            initial_design_numdata = bayopt_n_rounds,
                                                            exact_feval = False,
                                                            normalize_Y = True,
                                                            num_cores = multiprocessing.cpu_count()-1)
            print("Optimization:\n")
            Bayes_opt.run_optimization(max_iter=bayopt_n_rounds)
            best_arch = Bayes_opt.x_opt
        else:
            best_arch = [lstmunits_ref, denseunits_ref, embedding_ref, batch_size_ref, alpha_ref]
            
        print("\nThe architecture for this datatset is:\n\tLSTM units: {}\n\tDense units: {}\n\tEmbedding dimensions {}".\
             format(int(best_arch[0]), int(best_arch[1]), int(best_arch[2])))
        print("\tBatch size: {0:}\n\tLearning rate: 10^-({1:.1f})\n".format(int(best_arch[3]), float(best_arch[4])))
        
        print("***Training of the best model.***\n")
        # Train the model and predict
        K.clear_session()   
        # Define the multi-gpus model if necessary
        if n_gpus > 1:
            if bridge_type == 'NVLink':
                model = LSTMAttModel.create(inputtokens = max_length+1, 
                                            vocabsize = vocab_size, 
                                            lstmunits= int(best_arch[0]), 
                                            denseunits = int(best_arch[1]), 
                                            embedding = int(best_arch[2]))
            else:
                with tf.device('/cpu'):
                    model = LSTMAttModel.create(inputtokens = max_length+1, 
                                                vocabsize = vocab_size, 
                                                lstmunits= int(best_arch[0]), 
                                                denseunits = int(best_arch[1]), 
                                                embedding = int(best_arch[2]))
            print("Best model summary:\n")
            print(model.summary())
            print("\n")
            multi_model = ModelMGPU(model, gpus=n_gpus, bridge_type=bridge_type)
        else:
            model = LSTMAttModel.create(inputtokens = max_length+1, 
                                        vocabsize = vocab_size, 
                                        lstmunits= int(best_arch[0]), 
                                        denseunits = int(best_arch[1]), 
                                        embedding = int(best_arch[2]))

            print("Best model summary:\n")
            print(model.summary())
            print("\n")
            multi_model = model

        batch_size = int(best_arch[3])
        custom_adam = Adam(lr=math.pow(10,-float(best_arch[4])))
        # Compile the model
        multi_model.compile(loss="mse", optimizer=custom_adam, metrics=[metrics.mae,metrics.mse])
        
        # Checkpoint, Early stopping and callbacks definition
        filepath=save_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(selection_seed)+'.hdf5'
        
        checkpoint = ModelCheckpoint(filepath, 
                                     monitor='val_loss', 
                                     verbose=0, 
                                     save_best_only=True, 
                                     mode='min')

        earlystopping = EarlyStopping(monitor='val_loss', 
                                      min_delta=0, 
                                      patience=patience, 
                                      verbose=0, 
                                      mode='min')
                
        callbacks_list = [checkpoint, earlystopping]

        # Fit the model
        history = multi_model.fit_generator(generator = DataSequence(x_train_enum_tokens,
                                                                     vocab = tokens, 
                                                                     max_length = max_length, 
                                                                     props_set = y_train_enum, 
                                                                     batch_size = batch_size), 
                                            validation_data = DataSequence(x_valid_enum_tokens,
                                                                           vocab = tokens, 
                                                                           max_length = max_length, 
                                                                           props_set = y_valid_enum, 
                                                                           batch_size = min(len(x_valid_enum_tokens), batch_size)),
                                            epochs = n_epochs, 
                                            shuffle = True,
                                            initial_epoch = 0, 
                                            callbacks = callbacks_list)

        # Summarize history for losses per epoch
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(save_dir+'History_fit_LSTMAtt_'+data_name+'_model_weights.best_seed_'+str(selection_seed)+'.png', bbox_inches='tight')
        plt.close()
        
        print("Best val_loss @ Epoch #{}\n".format(np.argmin(history.history['val_loss'])+1))

        print("***Predictions from the best model.***\n")
        model.load_weights(save_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(selection_seed)+'.hdf5')
#         model.save(save_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(selection_seed)+'.hdf5')  
        model.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

        # predict and compare for the training, validation and test sets
        x_train_enum_tokens_tointvec = int_vec_encode(tokenized_smiles_list = x_train_enum_tokens, 
                                                      max_length = max_length+1, 
                                                      vocab = tokens)
        x_valid_enum_tokens_tointvec = int_vec_encode(tokenized_smiles_list = x_valid_enum_tokens, 
                                                      max_length = max_length+1, 
                                                      vocab = tokens)
        x_test_enum_tokens_tointvec = int_vec_encode(tokenized_smiles_list = x_test_enum_tokens, 
                                                     max_length = max_length+1, 
                                                     vocab = tokens)

        y_pred_train = model.predict(x_train_enum_tokens_tointvec)
        y_pred_valid = model.predict(x_valid_enum_tokens_tointvec)
        y_pred_test = model.predict(x_test_enum_tokens_tointvec)

        # compute a mean per set of augmented SMILES
        y_pred_train_mean, _ = mean_median_result(x_train_enum_card, y_pred_train)
        y_pred_valid_mean, _ = mean_median_result(x_valid_enum_card, y_pred_valid)
        y_pred_test_mean, _ = mean_median_result(x_test_enum_card, y_pred_test)

        # inverse transform the scaling of the property and plot 'predictions VS observations'
        y_pred_VS_true = scaler.inverse_transform(y_train) - \
                         scaler.inverse_transform(y_pred_train_mean.reshape(-1,1))
        mae_train = np.mean(np.absolute(y_pred_VS_true))
        mse_train = np.mean(np.square(y_pred_VS_true))
        corrcoef_train = r2_score(scaler.inverse_transform(y_train), \
                                 scaler.inverse_transform(y_pred_train_mean.reshape(-1,1)))
        print("For the training set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_train, np.sqrt(mse_train), corrcoef_train))

        y_pred_VS_true = scaler.inverse_transform(y_valid) - \
                         scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1))
        mae_valid = np.mean(np.absolute(y_pred_VS_true))
        mse_valid = np.mean(np.square(y_pred_VS_true))
        corrcoef_valid = r2_score(scaler.inverse_transform(y_valid), \
                                 scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1)))
        print("For the validation set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_valid, np.sqrt(mse_valid), corrcoef_valid))

        y_pred_VS_true = scaler.inverse_transform(y_test) - \
                         scaler.inverse_transform(y_pred_test_mean.reshape(-1,1))
        mae_test = np.mean(np.absolute(y_pred_VS_true))
        mse_test = np.mean(np.square(y_pred_VS_true))
        corrcoef_test = r2_score(scaler.inverse_transform(y_test), \
                                 scaler.inverse_transform(y_pred_test_mean.reshape(-1,1)))
        print("For the test set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_test, np.sqrt(mse_test), corrcoef_test))

        # Plot the final result
        plt.scatter(scaler.inverse_transform(y_train), 
                    scaler.inverse_transform(y_pred_train_mean.reshape(-1,1)), label="Train")
        plt.scatter(scaler.inverse_transform(y_valid), 
                    scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1)), label="Validation")
        plt.scatter(scaler.inverse_transform(y_test), 
                    scaler.inverse_transform(y_pred_test_mean.reshape(-1,1)), label="Test")
        plt.plot([np.min(data.iloc[:,1]),np.max(data.iloc[:,1])],
                 [np.min(data.iloc[:,1]),np.max(data.iloc[:,1])], 
                 '--', color = 'r', alpha = 0.5)
        plt.xlabel('Observations '+data_units, fontsize = 12)
        plt.ylabel('Predictions '+data_units, fontsize = 12)
        plt.legend()
        plt.savefig(save_dir+'TrainValidTest_Plot_LSTMAtt_'+data_name+'_model_weights.best_seed_'+str(selection_seed)+'.png', bbox_inches='tight')
        plt.close()
##