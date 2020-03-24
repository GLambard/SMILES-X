import os
import numpy as np
import matplotlib.pyplot as plt
import glob

from tensorflow.keras.models import load_model
from tensorflow.keras import metrics

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from itertools import cycle
from adjustText import adjust_text

from SMILESX import model, utils, token, augm

## Visualization of the Embedding layer 
# data: provided data (numpy array of: (SMILES, property))
# data_name: dataset's name
# k_fold_number: number of k-folds used for inference (Default: None, i.e. automatically detect k_fold_number from main.Main phase)
# k_fold_index: k-fold index to be used for visualization (Default: 0, i.e. first fold)
# augmentation: SMILES's augmentation (Default: False)
# indir: directory of already trained prediction models (*.hdf5) and vocabulary (*.txt) (Default: '../data/')
# outdir: directory for outputs (plots + .txt files) -> 'Embedding_Vis/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# affinity_propn: Affinity propagation tagging (Default: True)
# returns:
#         PCA visualization of a representation of SMILES tokens from the embedding layer

def Embedding_Vis(data, 
                  data_name, 
                  k_fold_number = None,
                  k_fold_index = 0,
                  augmentation = False, 
                  indir = "../data/", 
                  outdir = "../data/", 
                  affinity_propn = True, 
                  verbose = 0):
    
    if augmentation:
        p_dir_temp = 'Augm'
    else:
        p_dir_temp = 'Can'
        
    input_dir = indir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    save_dir = outdir+'Embedding_Vis/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
    
    for itype in ["txt","hdf5"]:
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
                print("The input directory does not contain any vocabulary (*_Vocabulary.txt file).\n")
            return
        
    if k_fold_index >= k_fold_number:
        print("***Process of inference automatically aborted!***")
        print("The condition \"0 <= k_fold_index < k_fold_number\" is not respected.\n")
        return
    
    print("****************************************************")
    print("***SMILES_X for embedding visualization starts...***")
    print("****************************************************\n")
    # Setting up the cross_validation on k-folds
    kf = KFold(n_splits=k_fold_number, random_state=123, shuffle=True)
    data_smiles = data.smiles.values
    data_prop = data.iloc[:,1].values.reshape(-1,1)
    kf.get_n_splits(data_smiles)
    for ifold, (train_index, valid_test_index) in enumerate(kf.split(data_smiles)):
        
        if ifold != k_fold_index:
            continue
        
        print("{}-fold initiated.".format(k_fold_index))

        print("Splitting of the dataset.")
        # Reproducing the data split of the requested fold (k_fold_index)
        x_train, _, _, y_train, _, _, _, _, _, _ = \
        utils.split_standardize(smiles_input = data_smiles, 
                                prop_input = data_prop, 
                                train_index = train_index, 
                                valid_test_index = valid_test_index)

        # data augmentation or not
        if augmentation == True:
            print("Data augmentation required.\n")
            canonical = False
            rotation = True
        else:
            print("No data augmentation required.\n")
            canonical = True
            rotation = False

        x_train_enum, _, _ = augm.Augmentation(x_train, y_train, canon=canonical, rotate=rotation)

        print("Number of enumerated SMILES from the training set: {}.".format(x_train_enum.shape[0]))

        print("Tokenization of SMILES from the training set.")
        # Tokenize SMILES from the training set
        x_train_enum_tokens = token.get_tokens(x_train_enum)

        train_unique_tokens = list(token.extract_vocab(x_train_enum_tokens))
        print("Tokens from the training set: {}".format(train_unique_tokens))
        print("Number of tokens only present in the training set: {}\n".format(len(train_unique_tokens)))
        train_unique_tokens.insert(0,'pad')

        # All tokens as a list
        tokens = token.get_vocab(input_dir+data_name+'_Vocabulary.txt')
        vocab_size = len(tokens)
        # Add 'pad', 'unk' tokens to the existing list
        tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)

        print("Full vocabulary (\"train+valid+test\" tokens): {}, of size: {}\n".format(tokens, vocab_size))

        # Load a trained prediction model
        model_train = load_model(input_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(k_fold_index)+'.hdf5', 
                                 custom_objects={'AttentionM': model.AttentionM()})

        print("PCA on the {}-fold model's embedding of all the tokens.".format(k_fold_index))
        print("(Tokens from (circles) and out of (crosses) the training set are shown. Colors distinguish clusters by computed affinity propagation.)")
    #    model_train.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

        model_embed_weights = model_train.layers[1].get_weights()[0]
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

        plt.savefig(save_dir+'Visualization_'+data_name+'_Embedding_fold_'+str(k_fold_index)+'.png', bbox_inches='tight')
        plt.show()
        
        print("\n************************************************")
        print("***SMILES_X for embedding visualization done.***")
        print("************************************************\n")
##
