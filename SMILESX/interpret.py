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

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import metrics

from SMILESX import utils, model, token, augm, main, inference

from pickle import load

from PIL import Image

## Interpretation of the SMILESX predictions
# data: provided data (dataframe of: (SMILES, property))
# data_name: dataset's name
# data_units: property's SI units
# k_fold_number: number of k-folds used for inference (Default: None, i.e. automatically detect k_fold_number from main.Main phase)
# k_fold_index: k-fold index to be used for visualization (Default: None, i.e. use all the models, then average)
# augmentation: SMILES's augmentation (Default: False)
# indir: directory of already trained prediction models (*.hdf5) and vocabulary (*.txt) (Default: '../data/')
# outdir: directory for outputs (plots + .txt files) -> 'Interpretation/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# font_size: font's size for writing SMILES tokens (Default: 15)
# font_rotation: font's orientation (Default: 'horizontal')
# returns:
#         The 1D and 2D attention maps 
#             The redder and darker the color is, 
#             the stronger is the attention on a given token. 
#         The temporal relative distance Tdist 
#             The closer to zero is the distance value, 
#             the closer is the temporary prediction on the SMILES fragment to the whole SMILES prediction.
class Interpretation:

    def __init__(self, 
                 data, 
                 data_name, 
                 data_units = '',
                 k_fold_number = None,
                 k_fold_index = None,
                 augmentation = False, 
                 indir = "../data/", 
                 outdir = "../data/", 
                 font_size = 15, 
                 font_rotation = 'vertical'):
    
        self.data = data
        self.data_name = data_name
        self.data_units = data_units
        self.k_fold_number = k_fold_number
        self.k_fold_index = k_fold_index
        self.augmentation = augmentation
        self.indir = indir
        self.outdir = outdir
        self.font_size = font_size
        self.font_rotation = font_rotation
        
        self.Inference_class = inference.Inference(data_name = self.data_name, 
                                                   data_units = self.data_units,
                                                   k_fold_number = self.k_fold_number,
                                                   augmentation = self.augmentation, 
                                                   indir = self.indir, 
                                                   outdir = self.outdir, 
                                                   return_attention = True)
        
        if augmentation:
            p_dir_temp = 'Augm'
        else:
            p_dir_temp = 'Can'

        self.input_dir = self.indir+'Main/'+'{}/{}/'.format(self.data_name,p_dir_temp)
        self.save_dir = self.outdir+'Interpretation/'+'{}/{}/'.format(self.data_name,p_dir_temp)
        
        if self.k_fold_index is not None and self.k_fold_number is not None:
       # smiles_toviz: targeted SMILES to visualize (Default: 'CCC')     if self.k_fold_index >= self.k_fold_number:
                print("***Process of inference automatically aborted!***")
                print("The condition \"0 <= k_fold_index < k_fold_number\" is not respected.\n")
                return
    
    # smiles_list: targeted list of SMILES to visualize (Default: ['CC','CCC','C=O'])
    # num_precision: numerical precision for displaying the predicted values (Default: 2)
    def interpret(self, smiles_list = ['CC','CCC','C=O'], num_precision = 2):

        print("************************************")
        print("***SMILES_X Interpreter starts...***")
        print("************************************\n")

#         # Check the submitted SMILES
#         mol_toviz = Chem.MolFromSmiles(smiles_toviz)
#         if mol_toviz != None:
#             smiles_toviz_can = Chem.MolToSmiles(mol_toviz)
#         else:
#             print("***Process of visualization automatically aborted!***")
#             print("The submitted SMILES is incorrect and cannot be sanitized by RDKit.\n")
#             return

#         smiles_toviz_x = np.array([smiles_toviz_can])
#         if smiles_toviz_can in np.array(data.smiles):
#             smiles_toviz_y = np.array([[data.iloc[np.where(data.smiles == smiles_toviz_x[0])[0][0],1]]])
#             print("The submitted SMILES is found in the dataset used for training the models.\n")
#         else:
#             smiles_toviz_y = np.array([[np.nan]])

#         # data augmentation or not
#         if augmentation == True:
#             print("***Data augmentation is required.***\n")
#             canonical = False
#             rotation = True
#         else:
#             print("***No data augmentation is required.***\n")
#             canonical = True
#             rotation = False

#         smiles_toviz_x_enum, smiles_toviz_x_enum_card, smiles_toviz_y_enum = \
#         augm.Augmentation(smiles_toviz_x, smiles_toviz_y, canon=canonical, rotate=rotation)

#         # Submitted SMILES tokenization 
#         smiles_toviz_x_enum_tokens = token.get_tokens(smiles_toviz_x_enum)

#         # Models ensembling
#         for ifold in range(k_fold_number):

#             if k_fold_index is not None:
#                 if ifold != k_fold_index:
#                     continue

#             # Load the scaler
#             scaler = load(open('scaler_fold_' + str(ifold) + '.pkl', 'rb'))

#             # Best architecture to visualize from
#             model_topredict = load_model(input_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(ifold)+'.hdf5', 
#                                                   custom_objects={'AttentionM': model.AttentionM()})
#             best_arch = [model_topredict.layers[2].output_shape[-1]/2, 
#                          model_topredict.layers[3].output_shape[-1], 
#                          model_topredict.layers[1].output_shape[-1]]

#             if ifold == 0:
#                 # Maximum of length of SMILES to process
#                 max_length = model_topredict.layers[0].output_shape[-1][1]
#                 smiles_toviz_x_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list= smiles_toviz_x_enum_tokens, 
#                                                                            max_length = max_length,
#                                                                            vocab = tokens)
#                 intermediate_output_array = np.empty(shape=(0, smiles_toviz_x_enum_card[0], max_lengh, 1), dtype='float')
#                 smiles_y_pred_mean_array = np.empty(shape=(0,len(smiles_checked)), dtype='float')

#             # Architecture to return attention weights
#             model_att = model.LSTMAttModel.create(inputtokens = max_length, 
#                                                   vocabsize = vocab_size, 
#                                                   lstmunits= int(best_arch[0]), 
#                                                   denseunits = int(best_arch[1]), 
#                                                   embedding = int(best_arch[2]), 
#                                                   return_proba = True)

#             model_att.load_weights(input_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(k_fold_index)+'.hdf5')
#     #    model_att.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

#             intermediate_layer_model = Model(inputs=model_att.input,
#                                              outputs=model_att.layers[-2].output)
#             intermediate_output = intermediate_layer_model.predict(smiles_toviz_x_enum_tokens_tointvec)

#             intermediate_output_array = np.append(intermediate_output_array, intermediate_output.reshape((1,)+intermediate_output.shape), axis = 0)

#         intermediate_output = np.mean(intermediate_output_array, axis = 0)
        
        # Output shapes:
        # Predictions: (batch_size,)
        # attention maps: (batch_size, max_length)
        # List of list of tokens: (batch_size) x (length of tokenized SMILES with padding)
        y_pred_dataframe, att_map_mean, att_map_std, smiles_x_tokens = \
            self.Inference_class.infer(smiles_list, check_smiles = True, return_att = True)
        smiles_x_retrieval = [''.join(ismiles[1:-1]) for ismiles in smiles_x_tokens]
        smiles_list_len = len(smiles_x_retrieval)
        smiles_x_tokens_len = [len(ismiles) for ismiles in smiles_x_tokens]
        y_true = np.zeros((smiles_list_len,), dtype='float')
        for ismiles, smiles_tmp in enumerate(smiles_x_retrieval):
            if smiles_tmp in self.data.smiles.values.tolist():
                y_true[ismiles] = self.data.iloc[np.where(self.data.smiles == smiles_tmp)[0][0],1]
            else:
                y_true[ismiles] = np.nan

#         smiles_toviz_x_card_cumsum_viz = np.cumsum(smiles_toviz_x_enum_card)
#         smiles_toviz_x_card_cumsum_shift_viz = shift(smiles_toviz_x_card_cumsum_viz, 1, cval=0)

#         mols_id = 0
#         ienumcard = smiles_toviz_x_card_cumsum_shift_viz[mols_id]

#         smiles_len_tmp = len(smiles_toviz_x_enum_tokens[ienumcard])
#         intermediate_output_tmp = intermediate_output[ienumcard,-smiles_len_tmp+1:-1].flatten().reshape(1,-1)
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        for ismiles in range(smiles_list_len): 
            
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
            fig.suptitle('SMILES: {}'.format(smiles_x_retrieval[ismiles]), fontsize = 16)
            
            # Predictions
            value_toprint = "Predicted value: {1:.{0}f} +/- {2:.{0}f} {3}".format(num_precision, 
                                                                                  y_pred_dataframe.iloc[ismiles,1], 
                                                                                  y_pred_dataframe.iloc[ismiles,2], 
                                                                                  self.data_units)
            value_toprint += "\nExp/Sim value: {1:.{0}f} {2}".format(num_precision, 
                                                                     y_true[ismiles], 
                                                                     self.data_units)
            
            axes[0,0].text(0.5, 0.5, value_toprint, 
                                 transform=axes[0,0].transAxes, 
                                 fontsize=self.font_size,
                                 verticalalignment='top', bbox=props)
            axes[0,0].axis('off')
            
            smiles_tmp = smiles_x_tokens[ismiles]
            
            # 1D attention map
            smiles_len_tmp = smiles_x_tokens_len[ismiles]
            axes[0,1].matshow(att_map_mean[ismiles,-smiles_len_tmp+1:-1].flatten().reshape(1,-1), cmap='Reds') # SMILES padding excluded
            axes[0,1].tick_params(axis='x', bottom = False)
            axes[0,1].set_xticks(np.arange(smiles_len_tmp-2))
            axes[0,1].set_xticklabels(smiles_tmp[1:-1], fontsize = self.font_size, rotation = self.font_rotation)
            axes[0,1].set_yticks([])
            #plt.savefig(save_dir+'Interpretation_1D_'+data_name+'_fold_'+str(k_fold_index)+'.png', bbox_inches='tight')

            # 2D attention map
            mol_tmp = Chem.MolFromSmiles(smiles_x_retrieval[ismiles])
            mol_df_tmp = pd.DataFrame([smiles_tmp[1:-1], att_map_mean[ismiles].\
                                       flatten().\
                                       tolist()[-smiles_len_tmp+1:-1]]).transpose()
            bond = ['-','=','#','$','/','\\','.','(',')']
            mol_df_tmp = mol_df_tmp[~mol_df_tmp.iloc[:,0].isin(bond)]
            mol_df_tmp = mol_df_tmp[[not itoken.isdigit() for itoken in mol_df_tmp.iloc[:,0].values.tolist()]]

            minmaxscaler = MinMaxScaler(feature_range=(0,1))
            norm_weights = minmaxscaler.fit_transform(mol_df_tmp.iloc[:,1].values.reshape(-1,1)).flatten().tolist()
            fig_tmp = GetSimilarityMapFromWeights(mol=mol_tmp, 
                                              size = (250,250), 
                                              scale=-1,  
                                              sigma=0.05,
                                              weights=norm_weights, 
                                              colorMap='Reds', 
                                              contourLines = 10,
                                              alpha = 0.25)
            fig_tmp.canvas.draw()
            axes[1,0].imshow(np.array(fig_tmp.canvas.renderer.buffer_rgba()))
            #fig.savefig(save_dir+'Interpretation_2D_'+data_name+'_fold_'+str(k_fold_index)+'.png', bbox_inches='tight')

    #     diff_topred_list = list()
    #     diff_totrue_list = list()
    #     for csubsmiles in range(1,smiles_len_tmp):
    #         isubsmiles = smiles_toviz_x_enum_tokens[ienumcard][:csubsmiles]+[' ']
    #         isubsmiles_tointvec= token.int_vec_encode(tokenized_smiles_list = [isubsmiles], 
    #                                                   max_length = max_length+1, 
    #                                                   vocab = tokens)
    #         predict_prop_tmp = model_topredict.predict(isubsmiles_tointvec)[0,0]
    #         diff_topred_tmp = (predict_prop_tmp-y_pred_test_tmp)/np.abs(y_pred_test_tmp)
    #         diff_topred_list.append(diff_topred_tmp)
    #         diff_totrue_tmp = (predict_prop_tmp-y_test_tmp)/np.abs(y_test_tmp)
    #         diff_totrue_list.append(diff_totrue_tmp)
    #     max_diff_topred_tmp = np.max(diff_topred_list)
    #     max_diff_totrue_tmp = np.max(diff_totrue_list)

    #     plt.figure(figsize=(15,7))
    #     markers, stemlines, baseline = plt.stem([ix for ix in range(smiles_len_tmp-1)], 
    #                                             diff_topred_list, 
    #                                             'k.-', 
    #                                              use_line_collection=True)
    #     plt.setp(baseline, color='k', linewidth=2, linestyle='--')
    #     plt.setp(markers, linewidth=1, marker='o', markersize=10, markeredgecolor = 'black')
    #     plt.setp(stemlines, color = 'k', linewidth=0.5, linestyle='-')
    #     plt.xticks(range(smiles_len_tmp-1), 
    #                smiles_toviz_x_enum_tokens[ienumcard][:-1],
    #                fontsize = font_size, 
    #                rotation = font_rotation)
    #     plt.yticks(fontsize = 20)
    #     plt.ylabel('Temporal relative distance', fontsize = 25, labelpad = 15)
    #     plt.savefig(save_dir+'Interpretation_temporal_'+data_name+'_fold_'+str(k_fold_index)+'.png', bbox_inches='tight')
##


def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w,h,4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis = 2)
    return buf

def fig2img (fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data (fig)
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring() )

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
