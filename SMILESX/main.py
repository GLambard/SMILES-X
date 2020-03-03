import numpy as np
import os
import math

import matplotlib.pyplot as plt

from numpy.random import seed
seed(12345)

import GPy, GPyOpt

from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
import tensorflow as tf
import multiprocessing

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from SMILESX import utils, token, augm, model

np.random.seed(seed=123)
np.set_printoptions(precision=3)

##
# To manage GPU usage and memory growth
# ngpus: number of GPUs to be used (Default: 1)
# gpus_list: list of GPU IDs to be used (Default: None), e.g. ['0','1','2']
# gpus_debug: print out the GPUs ongoing usage 
# If gpus_list and ngpus are both provided, gpus_list prevails
def set_gpuoptions(n_gpus = 1, gpus_list = None , gpus_debug = False):
    
    if gpus_list is not None:
        gpu_ids = ','.join(gpus_list)
    else:
        gpu_ids = ','.join([str(iid) for iid in range(ngpus)])
    # For fixing the GPU in use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    # The GPU id to use (e.g. "0", "1", etc.)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids);
        
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs detected and configured.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    gpus_list_len = len(gpus)
    if gpus_list_len > 0:
        if gpus_list_len > 1: 
            strategy = tf.distribute.MirroredStrategy()
        else: 
            strategy = tf.distribute.OneDeviceStrategy()
        print ('{} GPU devices will be used.'.format(strategy.num_replicas_in_sync))
        
        # To find out which devices your operations and tensors are assigned to
        tf.debugging.set_log_device_placement(gpus_debug)
        
        return strategy
    else:
        print("No GPU is detected in the system. SMILES-X needs at least one GPU to proceed.")
        return None
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
        batch_x = token.int_vec_encode(tokenized_smiles_list = tokenized_smiles_list_tmp, 
                                 max_length = max_length_tmp+1,
                                 vocab = self.vocab)
        #batch_x = self.batch[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.props_set[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)
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
# gpus_list: list of GPU IDs to be used (Default: None), e.g. ['0','1','2']
# gpus_debug: print out the GPUs ongoing usage 
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
         k_fold_number = 10, 
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
         gpus_list = None, 
         gpus_debug = False, 
         patience = 25, 
         n_epochs = 100):
    
    # GPUs options
    strategy = set_gpuoptions(n_gpus = n_gpus, 
                              gpus_list = gpus_list, 
                              gpus_debug = gpus_debug)
    if strategy is None:
        return
    ##
    
    # Data augmentation
    if augmentation:
        p_dir_temp = 'Augm'
        print("***Data augmentation to {}***\n".format(augmentation))
        canonical = False
        rotation = True
    else:
        p_dir_temp = 'Can'
        print("***No data augmentation has been required.***\n")
        canonical = True
        rotation = False
    ##
      
    # Output directory
    save_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)
    ##
        
    print("***SMILES_X starts...***\n\n")
    
    # Setting up the cross_validation on k-folds
    kf = KFold(n_splits=k_fold_number, random_state=123, shuffle=True)
    data_smiles = data.smiles.values
    data_prop = data.iloc[:,1].values.reshape(-1,1)
    kf.get_n_splits(data_smiles)
    for ifold, (train_index, valid_test_index) in enumerate(kf.split(data_smiles)):
        
        print("******")
        print("***Fold #{} initiated...***".format(ifold))
        print("******")
        
        print("***Splitting and standardization of the dataset.***\n")
        x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = \
        utils.split_standardize(smiles_input = data_smiles, 
                                prop_input = data_prop, 
                                train_index = train_index, 
                                valid_test_index = valid_test_index)
            
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
        x_train_enum_tokens = token.get_tokens(x_train_enum)
        x_valid_enum_tokens = token.get_tokens(x_valid_enum)
        x_test_enum_tokens = token.get_tokens(x_test_enum)
        
        print("Examples of tokenized SMILES from a training set:\n{}\n".\
        format(x_train_enum_tokens[:5]))
        
        # Vocabulary size computation
        all_smiles_tokens = x_train_enum_tokens+x_valid_enum_tokens+x_test_enum_tokens

        # Check if the vocabulary for current dataset exists already
        if os.path.exists(save_dir+data_name+'_Vocabulary.txt'):
            tokens = token.get_vocab(save_dir+data_name+'_Vocabulary.txt')
        else:
            tokens = token.extract_vocab(all_smiles_tokens)
            token.save_vocab(tokens, save_dir+data_name+'_Vocabulary.txt')
            tokens = token.get_vocab(save_dir+data_name+'_Vocabulary.txt')

        vocab_size = len(tokens)
        
        train_unique_tokens = token.extract_vocab(x_train_enum_tokens)
        print("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
        valid_unique_tokens = token.extract_vocab(x_valid_enum_tokens)
        print("Number of tokens only present in a validation set: {}".format(len(valid_unique_tokens)))
        print("Is the validation set a subset of the training set: {}".\
              format(valid_unique_tokens.issubset(train_unique_tokens)))
        print("What are the tokens by which they differ: {}\n".\
              format(valid_unique_tokens.difference(train_unique_tokens)))
        test_unique_tokens = token.extract_vocab(x_test_enum_tokens)
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
        
        # Add 'pad', 'unk' tokens to the existing list
        tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)
        
        # Maximum of length of SMILES to process
        max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
        print("Maximum length of tokenized SMILES: {} tokens (termination spaces included)\n".format(max_length))
        
        # predict and compare for the training, validation and test sets
        x_train_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = x_train_enum_tokens, 
                                                            max_length = max_length+1, 
                                                            vocab = tokens)
        x_valid_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = x_valid_enum_tokens, 
                                                            max_length = max_length+1, 
                                                            vocab = tokens)
        x_test_enum_tokens_tointvec = token.int_vec_encode(tokenized_smiles_list = x_test_enum_tokens, 
                                                           max_length = max_length+1, 
                                                           vocab = tokens)
        
        print("***Bayesian Optimization of the SMILESX's architecture.***\n")        
        if bayopt_on:
            # Operate the bayesian optimization of the neural architecture
            def create_mod(params):
                print('Model: {}'.format(params))

                model_tag = data_name

                K.clear_session()
                with strategy.scope():
                    model_opt = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                          vocabsize = vocab_size, 
                                                          lstmunits=int(params[:,0][0]), 
                                                          denseunits = int(params[:,1]), 
                                                          embedding = int(params[:,2][0]), 
                                                          seed = seed)

                    batch_size = int(params[:,3][0]) * strategy.num_replicas_in_sync
                    custom_adam = Adam(lr=math.pow(10,-float(params[:,4][0])))
                    model_opt.compile(loss='mse', optimizer=custom_adam, metrics=[metrics.mae,metrics.mse])
                    
                model_opt.predict()    
                    
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
        with strategy.scope():
            model_train = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                    vocabsize = vocab_size, 
                                                    lstmunits= int(best_arch[0]), 
                                                    denseunits = int(best_arch[1]), 
                                                    embedding = int(best_arch[2]))
            
            batch_size = int(best_arch[3]) * strategy.num_replicas_in_sync
            custom_adam = Adam(lr=math.pow(10,-float(best_arch[4])))
            # Compile the model
            model_train.compile(loss="mse", optimizer=custom_adam, metrics=[metrics.mae,metrics.mse])
            
        print("Best model summary:\n")
        print(model_train.summary())
        print("\n")
        
        # Checkpoint, Early stopping and callbacks definition
        filepath=save_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(ifold)+'.hdf5'
        
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
        history = model_train.fit_generator(generator = DataSequence(x_train_enum_tokens,
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
                                            callbacks = callbacks_list)

        # Summarize history for losses per epoch
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(save_dir+'History_fit_LSTMAtt_'+data_name+'_model_weights.best_fold_'+str(ifold)+'.png', bbox_inches='tight')
        plt.close()
        
        print("Best val_loss @ Epoch #{}\n".format(np.argmin(history.history['val_loss'])+1))

        print("***Predictions from the best model.***\n")
        model_train.load_weights(save_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(ifold)+'.hdf5')
        model_train.compile(loss="mse", optimizer='adam', metrics=[metrics.mae,metrics.mse])

        y_pred_train = model_train.predict(x_train_enum_tokens_tointvec)
        y_pred_valid = model_train.predict(x_valid_enum_tokens_tointvec)
        y_pred_test = model_train.predict(x_test_enum_tokens_tointvec)

        # compute a mean per set of augmented SMILES
        y_pred_train_mean, _ = utils.mean_median_result(x_train_enum_card, y_pred_train)
        y_pred_valid_mean, _ = utils.mean_median_result(x_valid_enum_card, y_pred_valid)
        y_pred_test_mean, _ = utils.mean_median_result(x_test_enum_card, y_pred_test)

        # inverse transform the scaling of the property and plot 'predictions VS observations'
        y_pred_VS_true_train = scaler.inverse_transform(y_train) - \
                               scaler.inverse_transform(y_pred_train_mean.reshape(-1,1))
        mae_train = np.mean(np.absolute(y_pred_VS_true_train))
        mse_train = np.mean(np.square(y_pred_VS_true_train))
        corrcoef_train = r2_score(scaler.inverse_transform(y_train), \
                                 scaler.inverse_transform(y_pred_train_mean.reshape(-1,1)))
        print("For the training set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_train, np.sqrt(mse_train), corrcoef_train))

        y_pred_VS_true_valid = scaler.inverse_transform(y_valid) - \
                               scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1))
        mae_valid = np.mean(np.absolute(y_pred_VS_true_valid))
        mse_valid = np.mean(np.square(y_pred_VS_true_valid))
        corrcoef_valid = r2_score(scaler.inverse_transform(y_valid), \
                                  scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1)))
        print("For the validation set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_valid, np.sqrt(mse_valid), corrcoef_valid))

        y_pred_VS_true_test = scaler.inverse_transform(y_test) - \
                              scaler.inverse_transform(y_pred_test_mean.reshape(-1,1))
        mae_test = np.mean(np.absolute(y_pred_VS_true_test))
        mse_test = np.mean(np.square(y_pred_VS_true_test))
        corrcoef_test = r2_score(scaler.inverse_transform(y_test), \
                                 scaler.inverse_transform(y_pred_test_mean.reshape(-1,1)))
        print("For the test set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_test, np.sqrt(mse_test), corrcoef_test))

        # Plot the final result
        # Unscaling the data
        y_train = scaler.inverse_transform(y_train)
        y_pred_train_mean = scaler.inverse_transform(y_pred_train_mean.reshape(-1,1))
        y_valid = scaler.inverse_transform(y_valid)
        y_pred_valid_mean = scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1))
        y_test = scaler.inverse_transform(y_test)
        y_pred_test_mean = scaler.inverse_transform(y_pred_test_mean.reshape(-1,1))

        # Changed colors, scaling and sizes
        plt.figure(figsize=(12, 8))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setting plot limits
        y_true_min = min(np.min(y_train), np.min(y_valid), np.min(y_test))
        y_true_max = max(np.max(y_train), np.max(y_valid), np.max(y_test))
        y_pred_min = min(np.min(y_pred_train_mean), np.min(y_pred_valid_mean), np.min(y_pred_test_mean))
        y_pred_max = max(np.max(y_pred_train_mean), np.max(y_pred_valid_mean), np.max(y_pred_test_mean))
        # Expanding slightly the canvas around the data points (by 10%)
        axmin = y_true_min-0.1*(y_true_max-y_true_min)
        axmax = y_true_max+0.1*(y_true_max-y_true_min)
        aymin = y_pred_min-0.1*(y_pred_max-y_pred_min)
        aymax = y_pred_max+0.1*(y_pred_max-y_pred_min)

        plt.xlim(min(axmin, aymin), max(axmax, aymax))
        plt.ylim(min(axmin, aymin), max(axmax, aymax))
                        
        plt.errorbar(y_train, 
                    y_pred_train_mean,
                    fmt='o',
                    label="Train",
                    elinewidth = 0, 
                    ms=5,
                    mfc='#519fc4',
                    markeredgewidth = 0,
                    alpha=0.7)
        plt.errorbar(y_valid,
                    y_pred_valid_mean,
                    elinewidth = 0,
                    fmt='o',
                    label="Validation", 
                    ms=5, 
                    mfc='#db702e',
                    markeredgewidth = 0,
                    alpha=0.7)
        plt.errorbar(y_test,
                    y_pred_test_mean,
                    elinewidth = 0,
                    fmt='o',
                    label="Test", 
                    ms=5, 
                    mfc='#cc1b00',
                    markeredgewidth = 0,
                    alpha=0.7)


        # Plot X=Y line
        plt.plot([max(plt.xlim()[0], plt.ylim()[0]), 
                  min(plt.xlim()[1], plt.ylim()[1])],
                 [max(plt.xlim()[0], plt.ylim()[0]), 
                  min(plt.xlim()[1], plt.ylim()[1])],
                 ':', color = '#595f69')
        
        plt.xlabel('Observations ' + data_units, fontsize = 12)
        plt.ylabel('Predictions ' + data_units, fontsize = 12)
        plt.legend()

        # Added fold number
        plt.savefig(save_dir+'TrainValid_Plot_LSTMAtt_'+data_name+'_model_weights.best_fold_'+str(ifold)+'.png', bbox_inches='tight', dpi=80)
        plt.close()
