       import os
import math
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import MinMaxScaler

from rdkit import Chem
from rdkit.Chem import Draw

from keras.models import Model
from keras.models import load_model
from keras import metrics
from keras import backend as K
import tensorflow as tf

from SMILESX import utils, model, token, augm

##
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

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
    utils.random_split(smiles_input=data.smiles, 
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
    augm.Augmentation(x_train, y_train, canon=canonical, rotate=rotation)

    x_valid_enum, x_valid_enum_card, y_valid_enum = \
    augm.Augmentation(x_valid, y_valid, canon=canonical, rotate=rotation)

    x_test_enum, x_test_enum_card, y_test_enum = \
    augm.Augmentation(x_test, y_test, canon=canonical, rotate=rotation)
    
    smiles_toviz_x_enum, smiles_toviz_x_enum_card, smiles_toviz_y_enum = \
    augm.Augmentation(smiles_toviz_x, smiles_toviz_y, canon=canonical, rotate=rotation)

    print("Enumerated SMILES:\n\tTraining set: {}\n\tValidation set: {}\n\tTest set: {}\n".\
    format(x_train_enum.shape[0], x_valid_enum.shape[0], x_test_enum.shape[0]))

    print("***Tokenization of SMILES.***\n")
    # Tokenize SMILES per dataset
    x_train_enum_tokens = token.get_tokens(x_train_enum)
    x_valid_enum_tokens = token.get_tokens(x_valid_enum)
    x_test_enum_tokens = token.get_tokens(x_test_enum)
    
    smiles_toviz_x_enum_tokens = token.get_tokens(smiles_toviz_x_enum)

    print("Examples of tokenized SMILES from a training set:\n{}\n".\
    format(x_train_enum_tokens[:5]))

    # Vocabulary size computation
    all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens
    tokens = token.extract_vocab(all_smiles_tokens)
    vocab_size = len(tokens)

    train_unique_tokens = list(token.extract_vocab(x_train_enum_tokens))
    print(train_unique_tokens)
    print("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
    train_unique_tokens.insert(0,'pad')
    
    # Tokens as a list
    tokens = token.get_vocab(input_dir+data_name+'_tokens_set_seed'+str(selection_seed)+'.txt')
    # Add 'pad', 'unk' tokens to the existing list
    tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)
    
    print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))

    # Maximum of length of SMILES to process
    max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
    print("Maximum length of tokenized SMILES: {} tokens\n".format(max_length))

    # Transformation of tokenized SMILES to vector of intergers and vice-versa
    token_to_int = token.get_tokentoint(tokens)
    int_to_token = token.get_inttotoken(tokens)

    # Best architecture to visualize from
    model_topredict = load_model(input_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(selection_seed)+'.hdf5', 
                                          custom_objects={'AttentionM': model.AttentionM()})
    best_arch = [model_topredict.layers[2].output_shape[-1]/2, 
                 model_topredict.layers[3].output_shape[-1], 
                 model_topredict.layers[1].output_shape[-1]]

    # Architecture to return attention weights
    model_att = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                          vocabsize = vocab_size, 
                                          lstmunits= int(best_arch[0]), 
                                          denseunits = int(best_arch[1]), 
                                          embedding = int(best_arch[2]), 
                                          return_proba = True)

    print("Best model summary:\n")
    print(model_att.summary())
    print("\n")

    print("***Interpretation from the best model.***\n")
    model_att.load_weights(input_dir+'LSTMAtt_'+data_name+'_model.best_seed_'+str(selection_seed)+'.hdf5')
    model_att.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

    smiles_toviz_x_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list= smiles_toviz_x_enum_tokens, 
                                                               max_length = max_length+1,
                                                               vocab = tokens)
    
    intermediate_layer_model = Model(inputs=model_att.input,
                                     outputs=model_att.layers[-2].output)
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
        isubsmiles_tointvec= token.int_vec_encode(tokenized_smiles_list = [isubsmiles], 
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
