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
                                                   k_fold_index = self.k_fold_index, 
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
    
    # smiles_list: targeted list of SMILES to visualize (Default: ['CC','CCC','C=O'])
    # num_precision: numerical precision for displaying the predicted values (Default: 2)
    def interpret(self, smiles_list = ['CC','CCC','C=O'], num_precision = 2):

        print("************************************")
        print("***SMILES_X Interpreter starts...***")
        print("************************************\n")
        
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
        
        for ismiles in range(smiles_list_len): 
            
            print("\n\n*******")
            print("SMILES: {}".format(smiles_x_retrieval[ismiles]))
            
            # Predictions
            value_toprint = "Predicted value: {1:.{0}f} +/- {2:.{0}f} {3}".format(num_precision, 
                                                                                  y_pred_dataframe.iloc[ismiles,1], 
                                                                                  y_pred_dataframe.iloc[ismiles,2], 
                                                                                  self.data_units)
            value_toprint += "\nExp/Sim value: {1:.{0}f} {2}\n".format(num_precision, 
                                                                       y_true[ismiles], 
                                                                       self.data_units)
            print(value_toprint)
            with open(self.save_dir+'smiles_metadata_'+str(ismiles)+'.txt','w') as f:
                f.write("SMILES: %s\n" % smiles_x_retrieval[ismiles])
                f.write("%s\n" % value_toprint)
            
            smiles_tmp = smiles_x_tokens[ismiles]
            smiles_len_tmp = smiles_x_tokens_len[ismiles]
            
            # 1D attention map
            plt.matshow(att_map_mean[ismiles,-smiles_len_tmp+1:-1].flatten().reshape(1,-1), cmap='Reds') # SMILES padding excluded
            plt.tick_params(axis='x', bottom = False)
            plt.xticks(np.arange(smiles_len_tmp-2), smiles_tmp[1:-1], fontsize = self.font_size, rotation = self.font_rotation)
            plt.yticks([])
            plt.savefig(self.save_dir+'smiles_interpretation1D_'+str(ismiles)+'.png', bbox_inches='tight')
            plt.show()

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
                                                  sigma=0.05,
                                                  weights=norm_weights, 
                                                  colorMap='Reds', 
                                                  alpha = 0.25)
            plt.savefig(self.save_dir+'smiles_interpretation2D_'+str(ismiles)+'.png', bbox_inches='tight')
            plt.show()

            # Temporal relative distance plot
            # Observation based on non-augmented SMILES because of SMILES sequential elongation
            plt.figure(figsize=(15,7))
            diff_topred_list = list()
            isubsmiles_list = list()
            for csubsmiles in range(2,smiles_len_tmp):
                isubsmiles_list.append(''.join(smiles_tmp[1:csubsmiles]))
            predict_prop_tmp = self.Inference_class.infer(isubsmiles_list, check_smiles = False, return_att = False)
            predict_prop_last = predict_prop_tmp.ens_pred_mean.values[-1]
            diff_topred_tmp = (predict_prop_tmp.ens_pred_mean.values-predict_prop_last)/np.abs(predict_prop_last)
            max_diff_topred_tmp = np.max(diff_topred_tmp)

            markers, stemlines, baseline = plt.stem([ix for ix in range(1,smiles_len_tmp-1)], 
                                                    diff_topred_tmp, 
                                                    'k.-', 
                                                     use_line_collection=True)
            plt.setp(baseline, color='k', linewidth=2, linestyle='--')
            plt.setp(markers, linewidth=1, marker='o', markersize=10, markeredgecolor = 'black')
            plt.setp(stemlines, color = 'k', linewidth=0.5, linestyle='-')
            plt.xticks(range(1,smiles_len_tmp-1), 
                       smiles_tmp[1:-1],
                       fontsize = self.font_size, 
                       rotation = self.font_rotation)
            plt.yticks(fontsize = 20)
            plt.ylabel('Cumulative SMILES path', fontsize = 20, labelpad = 15)
            plt.ylabel('Temporal relative distance', fontsize = 20, labelpad = 15)
            plt.savefig(self.save_dir+'smiles_interpretation_trd_'+str(ismiles)+'.png', bbox_inches='tight')
            plt.show()
            
        print("\n\n********************************")
        print("***SMILES_X Interpreter done.***")
        print("********************************\n")
##

## Attention weights depiction
# from https://github.com/rdkit/rdkit/blob/24f1737839c9302489cadc473d8d9196ad9187b4/rdkit/Chem/Draw/SimilarityMaps.py
# returns:
#         a similarity map for a molecule given the attention weights
def GetSimilarityMapFromWeights(mol, weights, colorMap=None, size=(250, 250),
                                sigma=0.05, coordScale=1.5, step=0.01, colors='k', contourLines=10,
                                alpha=0.5, **kwargs):
    """
    Generates the similarity map for a molecule given the atomic weights.
    Parameters:
    mol -- the molecule of interest
    colorMap -- the matplotlib color map scheme, default is custom PiWG color map
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
    x, y, z = Draw.calcAtomGaussians(mol, sigma, weights=weights, step=step)
    # scaling
    maxScale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
    minScale = min(math.fabs(np.min(z)), math.fabs(np.max(z)))
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
