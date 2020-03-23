import numpy as np
import os
import math
import glob

import tqdm

import matplotlib.pyplot as plt

from numpy.random import seed
seed(12345)

import GPy, GPyOpt

from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

import multiprocessing

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from SMILESX import utils, token, augm, model, clr_callback

import logging
import datetime
import time

np.random.seed(seed=123)
np.set_printoptions(precision=3)

##
# To manage GPU usage and memory growth
# ngpus: number of GPUs to be used (Default: 1)
# gpus_list: list of GPU IDs to be used (Default: None), e.g. ['0','1','2']
# gpus_debug: print out the GPUs ongoing usage 
# If gpus_list and ngpus are both provided, gpus_list prevails
def set_gpuoptions(n_gpus = 1, gpus_list = None , gpus_debug = False):
    
    # To find out which devices your operations and tensors are assigned to
    tf.debugging.set_log_device_placement(gpus_debug)
    
    if gpus_list is not None:
        gpu_ids = gpus_list
    else:
        gpu_ids = [str(iid) for iid in range(n_gpus)]
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
    
    gpus_list_len = len(logical_gpus)
    if gpus_list_len > 0:
        if gpus_list_len > 1: 
            strategy = tf.distribute.MirroredStrategy()
        else: 
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:"+gpu_ids[0])
        print ('{} GPU device(s) will be used.\n'.format(strategy.num_replicas_in_sync))
        
        return strategy, logical_gpus
    else:
        print("No GPU is detected in the system. SMILES-X needs at least one GPU to proceed.")
        return None
##

## Data sequence to be fed to the neural network during training through batches of data
class DataSequence(Sequence):
    # Initialization
    # smiles_set: array of tokenized SMILES of dimensions (number_of_SMILES, max_length)
    # props_set: array of targeted property
    # batch_size: batch's size
    # returns: 
    #         a batch of arrays of tokenized and encoded SMILES, 
    #         a batch of SMILES property
    def __init__(self, smiles_set, props_set, batch_size):
        self.smiles_set = smiles_set
        self.props_set = props_set
        self.batch_size = batch_size
        self.iepoch = 0

    def on_epoch_end(self):
        self.iepoch += 1
        
    def __len__(self):
        return int(np.ceil(len(self.smiles_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.smiles_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.props_set[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
##


## SMILESX main pipeline
# data: provided data (numpy array of: (SMILES, property))
# data_name: dataset's name
# bayopt_bounds: bounds contraining the Bayesian search of neural architectures
# data_units: property's SI units
# k_fold_number: number of k-folds used for cross-validation (Default: 8)
# augmentation: SMILES augmentation (Default: False)
# outdir: directory for outputs (plots + .txt files) -> 'Main/'+'{}/{}/'.format(data_name,p_dir_temp) is then created
# n_seeds: number of fixed seeds for initializing a neural architecture and averaging its predictions during Bayesian architecture search (Default:1)
# bayopt_n_rounds: number of architectures to be sampled during Bayesian architecture search (initialization + optimization) (Default: 25)
# bayopt_on: Use Bayesian architecture search or not (Default: True)
# lstmunits_ref: number of LSTM units for the k_fold_index if Bayesian architecture search is off
# denseunits_ref: number of dense units for the k_fold_index if Bayesian architecture search is off
# embedding_ref: number of embedding dimensions for the k_fold_index if Bayesian architecture search is off
# seed_ref: neural architecture's initialization seed (Default: None)
# n_gpus: number of GPUs to be used in parallel (Default: 1)
# gpus_list: list of GPU IDs to be used (Default: None), e.g. ['0','1','2']
# gpus_debug: print out the GPUs ongoing usage 
# patience: number of epochs to respect before stopping a training after minimal validation's error (Default: 25)
# n_epochs: maximum of epochs for training (Default: 1000)
# batchsize_pergpu: Batch size used per GPU (Default: None, i.e. self-defined according to the augmentation statistics)
# lr_schedule: learning rate schedule (Default: None), e.g. None, 'decay' (step decay) or 'clr' (cyclical)
# lr_min: maximum learning rate used during learning rate scheduling (Default: 1e-5)
# lr_max: minimum learning rate used during learning rate scheduling (Default: 1e-2)
# verbose: model fit verbosity (Default: 0), e.g. 0, 1 or 2
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
         n_seeds = 1, 
         bayopt_n_rounds = 25, 
         bayopt_on = True, 
         lstmunits_ref = 512, 
         denseunits_ref = 512, 
         embedding_ref = 512, 
         seed_ref = None, 
         n_gpus = 1, 
         gpus_list = None, 
         gpus_debug = False,
         patience = 25, 
         n_epochs = 100, 
         batchsize_pergpu = None, 
         lr_schedule = None, 
         lr_min = 1e-5, 
         lr_max = 1e-2, 
         verbose = 0):
    
    # GPUs options
    strategy, gpus = set_gpuoptions(n_gpus = n_gpus, 
                                    gpus_list = gpus_list, 
                                    gpus_debug = gpus_debug)
    if strategy is None:
        return
    ##
    
    # Data augmentation
    if augmentation:
        p_dir_temp = 'Augm'
        canonical = False
        rotation = True
    else:
        p_dir_temp = 'Can'
        canonical = True
        rotation = False
    ##
      
    # Output directory
    save_dir = outdir+'Main/'+'{}/{}/'.format(data_name,p_dir_temp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        for itype in ["png", "hdf5"]: # vocabulary *.txt and log files are kept
            exists_files = glob.glob(save_dir + "*." + itype)
            for ifile in exists_files:
                os.remove(ifile)
    ##
    
    # Logging
    currentDT = datetime.datetime.now()
    strDT = currentDT.strftime("%Y-%m-%d_%H:%M:%S")
    logging.basicConfig(filename=save_dir+strDT+'_Main.log', filemode='w', 
                        level=logging.INFO, format='%(message)s')
    ##
    
    logging.info("***Configuration parameters:***")
    logging.info("data = {}".format(data)) 
    logging.info("data_name = \'{}\'".format(data_name)) 
    logging.info("bayopt_bounds = {}".format(bayopt_bounds)) 
    logging.info("data_units = \'{}\'".format(data_units))
    logging.info("k_fold_number = {}".format(k_fold_number)) 
    logging.info("augmentation = {}".format(augmentation)) 
    logging.info("outdir = \'{}\'".format(outdir)) 
    logging.info("n_seeds = {}".format(n_seeds)) 
    logging.info("bayopt_n_rounds = {}".format(bayopt_n_rounds)) 
    logging.info("bayopt_on = {}".format(bayopt_on)) 
    logging.info("lstmunits_ref = {}".format(lstmunits_ref)) 
    logging.info("denseunits_ref = {}".format(denseunits_ref)) 
    logging.info("embedding_ref = {}".format(embedding_ref)) 
    logging.info("seed_ref = {}".format(seed_ref))
    logging.info("n_gpus = {}".format(n_gpus)) 
    logging.info("gpus_list = {}".format(gpus_list)) 
    logging.info("gpus_debug = {}".format(gpus_debug))
    logging.info("patience = {}".format(patience)) 
    logging.info("n_epochs = {}".format(n_epochs))
    logging.info("batchsize_pergpu = {}".format(batchsize_pergpu)) 
    logging.info("lr_schedule = {}".format(lr_schedule)) 
    logging.info("lr_min = {}".format(lr_min))
    logging.info("lr_max = {}".format(lr_max))
    logging.info("verbose = {}".format(verbose))
    logging.info("******\n")
    
    logging.info("{} Logical GPUs detected and configured.\n".format(len(gpus)))
    
    if augmentation:
        logging.info("***Data augmentation to {}***\n\n".format(augmentation))
    else:
        logging.info("***No data augmentation has been required.***\n\n")
    
    logging.info("************************")
    logging.info("***SMILES_X starts...***")
    logging.info("************************\n\n")
    print("***SMILES_X starts...***\n")
    print("The SMILES_X process can be followed in the "+save_dir+strDT+'_Main.log'+" file.\n")
    # Setting up the seeds for models initialization
    seed_list = np.random.randint(int(1e6), size = n_seeds).tolist()
    # Setting up the scores summary
    scores_summary = {'train': [], 
                      'valid': [], 
                      'test': []}
    # Setting up the cross_validation on k-folds
    kf = KFold(n_splits=k_fold_number, random_state=123, shuffle=True)
    data_smiles = data.smiles.values
    data_prop = data.iloc[:,1].values.reshape(-1,1)
    kf.get_n_splits(data_smiles)
    for ifold, (train_index, valid_test_index) in enumerate(kf.split(data_smiles)):
        
        if ifold == 0:
            start_time = time.time()
            print("Processing the 1st fold of data...", end = '\r')
        elif ifold > 0 and ifold < (k_fold_number-1):
            if ifold == 1:
                onefold_time = (time.time() - start_time)
            print("Remaining time: {:.2f} h. Processing fold #{} of data...".format((k_fold_number-ifold)*onefold_time/3600., ifold), end = '\r')
        else:
            if ifold == 1:
                onefold_time = (time.time() - start_time)
            print("Remaining time: <{:.2f} h. Processing the last fold of data...\n".format(onefold_time/3600.))
        
        logging.info("******")
        logging.info("***Fold #{} initiated...***".format(ifold))
        logging.info("******")
        
        logging.info("***Splitting and standardization of the dataset.***\n")
        x_train, x_valid, x_test, y_train, y_valid, y_test, scaler, y_train_unscaled, y_valid_unscaled, y_test_unscaled = \
        utils.split_standardize(smiles_input = data_smiles, 
                                prop_input = data_prop, 
                                train_index = train_index, 
                                valid_test_index = valid_test_index, 
                                logger = logging.getLogger(__name__))
            
        x_train_enum, x_train_enum_card, y_train_enum = \
        augm.Augmentation(x_train, y_train, canon=canonical, rotate=rotation)

        x_valid_enum, x_valid_enum_card, y_valid_enum = \
        augm.Augmentation(x_valid, y_valid, canon=canonical, rotate=rotation)

        x_test_enum, x_test_enum_card, y_test_enum = \
        augm.Augmentation(x_test, y_test, canon=canonical, rotate=rotation)
        
        logging.info("Enumerated SMILES:\n\tTraining set: {}\n\tValidation set: {}\n\tTest set: {}\n".\
        format(x_train_enum.shape[0], x_valid_enum.shape[0], x_test_enum.shape[0]))
        
        logging.info("***Tokenization of SMILES.***\n")
        # Tokenize SMILES per dataset
        x_train_enum_tokens = token.get_tokens(x_train_enum)
        x_valid_enum_tokens = token.get_tokens(x_valid_enum)
        x_test_enum_tokens = token.get_tokens(x_test_enum)
        
        logging.info("Examples of tokenized SMILES from a training set:\n{}\n".\
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
        logging.info("Number of tokens only present in a training set: {}\n".format(len(train_unique_tokens)))
        valid_unique_tokens = token.extract_vocab(x_valid_enum_tokens)
        logging.info("Number of tokens only present in a validation set: {}".format(len(valid_unique_tokens)))
        logging.info("Is the validation set a subset of the training set: {}".\
              format(valid_unique_tokens.issubset(train_unique_tokens)))
        logging.info("What are the tokens by which they differ: {}\n".\
              format(valid_unique_tokens.difference(train_unique_tokens)))
        test_unique_tokens = token.extract_vocab(x_test_enum_tokens)
        logging.info("Number of tokens only present in a test set: {}".format(len(test_unique_tokens)))
        logging.info("Is the test set a subset of the training set: {}".\
              format(test_unique_tokens.issubset(train_unique_tokens)))
        logging.info("What are the tokens by which they differ: {}".\
              format(test_unique_tokens.difference(train_unique_tokens)))
        logging.info("Is the test set a subset of the validation set: {}".\
              format(test_unique_tokens.issubset(valid_unique_tokens)))
        logging.info("What are the tokens by which they differ: {}\n".\
              format(test_unique_tokens.difference(valid_unique_tokens)))
        
        logging.info("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))
        
        # Add 'pad', 'unk' tokens to the existing list
        tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)
        
        # Maximum of length of SMILES to process
        max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
        logging.info("Maximum length of tokenized SMILES: {} tokens (termination spaces included)\n".format(max_length))
        
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
        
        logging.info("***Bayesian Optimization of the SMILESX's architecture.***\n") 
        
        if batchsize_pergpu is None:
            batch_size_list = np.array([int(2**itn) for itn in range(3,11)])
            batchsize_pergpu = batch_size_list[np.argmax((batch_size_list // np.max(x_train_enum_card)) == 1.)]
        batch_size = batchsize_pergpu * strategy.num_replicas_in_sync
        
        if bayopt_on:
            # Operate the bayesian optimization of the neural architecture
            def create_mod(params):
                params = params.astype(int).flatten().tolist()
                logging.info('Model: {}'.format(params))

                K.clear_session()
                if gpus:
                    mse_train_tmp = []
                    for iseed in seed_list:
                        y_pred_train_tmp = None
                        with tf.device(gpus[0].name):
                            model_opt = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                                  vocabsize = vocab_size, 
                                                                  lstmunits = params[0], 
                                                                  denseunits = params[1], 
                                                                  embedding = params[2], 
                                                                  seed = iseed)
                            y_pred_train_tmp = model_opt.predict(x_train_enum_tokens_tointvec, batch_size//strategy.num_replicas_in_sync)
                        with tf.device('/CPU:0'):
                            y_pred_train_mean_tmp, _ = utils.mean_median_result(x_train_enum_card, y_pred_train_tmp)  
                            y_pred_VS_true_train_tmp = y_train - y_pred_train_mean_tmp.reshape(-1,1)
                            mse_train_tmp.append(np.mean(np.square(y_pred_VS_true_train_tmp)))
                    mse_train_mean_tmp = np.mean(mse_train_tmp)
                    mse_train_std_tmp = np.std(mse_train_tmp)
                else:
                    logging.warning("Physical GPU(s) list doesn't exist.")
                    
                if math.isnan(mse_train_mean_tmp): # discard diverging architectures (rare event)
                    mse_train_mean_tmp = math.inf
                    mse_train_std_tmp = math.inf
                logging.info('Train MSE mean: {0:0.4f}, MSE std: {1:0.4f}'.format(mse_train_mean_tmp, mse_train_std_tmp))

                return mse_train_mean_tmp

            logging.info("Random initialization:\n")
            Bayes_opt = GPyOpt.methods.BayesianOptimization(f=create_mod, 
                                                            domain=bayopt_bounds, 
                                                            acquisition_type = 'EI',
                                                            acquisition_jitter = 0.1, 
                                                            initial_design_numdata = bayopt_n_rounds,
                                                            exact_feval = True,
                                                            normalize_Y = False,
                                                            num_cores = 1) #multiprocessing.cpu_count()-1
            logging.info("\nOptimization:\n")
            Bayes_opt.run_optimization(max_iter=bayopt_n_rounds)
            best_arch = Bayes_opt.x_opt.astype(int).tolist()
            
            # Find best seed
            if len(seed_list) > 1:
                K.clear_session()
                if gpus:
                    mse_train_tmp = []
                    for iseed in seed_list:
                        y_pred_train_tmp = None
                        with tf.device(gpus[0].name):
                            model_opt = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                                  vocabsize = vocab_size, 
                                                                  lstmunits= best_arch[0], 
                                                                  denseunits = best_arch[1], 
                                                                  embedding = best_arch[2], 
                                                                  seed = iseed)
                            y_pred_train_tmp = model_opt.predict(x_train_enum_tokens_tointvec, batch_size//strategy.num_replicas_in_sync)
                        with tf.device('/CPU:0'):
                            y_pred_train_mean_tmp, _ = utils.mean_median_result(x_train_enum_card, y_pred_train_tmp)  
                            y_pred_VS_true_train_tmp = y_train - y_pred_train_mean_tmp.reshape(-1,1)
                            mse_train_tmp.append(np.mean(np.square(y_pred_VS_true_train_tmp)))
                    best_seed = seed_list[np.argmin(mse_train_tmp)]
                else:
                    logging.warning("Physical GPU(s) list doesn't exist.")
            else:
                best_seed = seed_list[0]
            
            best_arch = best_arch + [best_seed]
        else:
            best_arch = [lstmunits_ref, denseunits_ref, embedding_ref, seed_ref]
            
        logging.info("\nThe best architecture for this datatset is:\n\tLSTM units: {}\n\tDense units: {}\n\tEmbedding dimensions {}\n".\
             format(best_arch[0], best_arch[1], best_arch[2]))
        
        logging.info("***Training of the best model.***\n")
        # Train the model and predict
        K.clear_session()   
        with strategy.scope():
            model_train = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                    vocabsize = vocab_size, 
                                                    lstmunits= best_arch[0], 
                                                    denseunits = best_arch[1], 
                                                    embedding = best_arch[2], 
                                                    seed = best_arch[3])            
            model_train.compile(loss="mse", optimizer=Adam(), metrics=[metrics.mae,metrics.mse])
        
        logging.info("Best model summary:")
        model_train.summary(print_fn=logging.info)
        logging.info("\n")
        
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
        
        schedule = utils.step_decay(initAlpha = 1e-2, finalAlpha = 1e-5, gamma = 0.95, epochs = n_epochs)
        
        clr = clr_callback.CyclicLR(base_lr = 1e-5, max_lr = 1e-2, 
                                    step_size = 8 * (x_train_enum_tokens_tointvec.shape[0] // (batch_size//strategy.num_replicas_in_sync)), 
                                    mode='triangular')
        
        if lr_schedule is None:
            callbacks_list = [checkpoint, earlystopping, 
                              utils.LoggingCallback(logging.info)] # default learning rate for Adam optimizer 
        elif lr_schedule == 'decay':
            callbacks_list = [checkpoint, earlystopping, LearningRateScheduler(schedule), 
                              utils.LoggingCallback(logging.info)] # learning rate step decay 
        elif lr_schedule == 'clr':
            callbacks_list = [checkpoint, earlystopping, clr, 
                              utils.LoggingCallback(logging.info)] # cyclical learning rate

        # Fit the model
        logging.info("Training:")
        history = model_train.fit(DataSequence(x_train_enum_tokens_tointvec,
                                               props_set = y_train_enum, 
                                               batch_size = batch_size), 
                                  validation_data = DataSequence(x_valid_enum_tokens_tointvec,
                                                                 props_set = y_valid_enum, 
                                                                 batch_size = min(len(x_valid_enum_tokens_tointvec), batch_size)),
                                  epochs = n_epochs, 
                                  shuffle = True,
                                  callbacks = callbacks_list, 
                                  verbose = verbose)

        # Summarize history for losses per epoch
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(save_dir+'History_fit_LSTMAtt_'+data_name+'_model_weights.best_fold_'+str(ifold)+'.png', bbox_inches='tight')
        plt.close()
        
        logging.info("\nBest val_loss @ Epoch #{}\n".format(np.argmin(history.history['val_loss'])+1))

        logging.info("***Prediction scores from the best model.***\n")
        with tf.device(gpus[0].name):
            K.clear_session()
            model_train = model.LSTMAttModel.create(inputtokens = max_length+1, 
                                                    vocabsize = vocab_size, 
                                                    lstmunits= best_arch[0], 
                                                    denseunits = best_arch[1], 
                                                    embedding = best_arch[2], 
                                                    seed = best_arch[3])
            model_train.load_weights(save_dir+'LSTMAtt_'+data_name+'_model.best_fold_'+str(ifold)+'.hdf5')
        
            y_pred_train = model_train.predict(x_train_enum_tokens_tointvec, batch_size//strategy.num_replicas_in_sync)
            y_pred_valid = model_train.predict(x_valid_enum_tokens_tointvec, batch_size//strategy.num_replicas_in_sync)
            y_pred_test = model_train.predict(x_test_enum_tokens_tointvec, batch_size//strategy.num_replicas_in_sync)

        # compute a mean per set of augmented SMILES
        y_pred_train_mean, _ = utils.mean_median_result(x_train_enum_card, y_pred_train)
        y_pred_valid_mean, _ = utils.mean_median_result(x_valid_enum_card, y_pred_valid)
        y_pred_test_mean, _ = utils.mean_median_result(x_test_enum_card, y_pred_test)

        # unscale prediction's outcomes
        y_pred_train_mean = scaler.inverse_transform(y_pred_train_mean.reshape(-1,1))
        y_pred_valid_mean = scaler.inverse_transform(y_pred_valid_mean.reshape(-1,1))
        y_pred_test_mean = scaler.inverse_transform(y_pred_test_mean.reshape(-1,1))
        
        # inverse transform the scaling of the property and plot 'predictions VS observations'
        y_pred_VS_true_train = y_train_unscaled - y_pred_train_mean
        mae_train = np.mean(np.absolute(y_pred_VS_true_train))
        rmse_train = np.sqrt(np.mean(np.square(y_pred_VS_true_train)))
        corrcoef_train = r2_score(y_train_unscaled, y_pred_train_mean)
        logging.info("For the training set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_train, rmse_train, corrcoef_train))

        y_pred_VS_true_valid = y_valid_unscaled - y_pred_valid_mean
        mae_valid = np.mean(np.absolute(y_pred_VS_true_valid))
        rmse_valid = np.sqrt(np.mean(np.square(y_pred_VS_true_valid)))
        corrcoef_valid = r2_score(y_valid_unscaled, y_pred_valid_mean)
        logging.info("For the validation set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_valid, rmse_valid, corrcoef_valid))

        y_pred_VS_true_test = y_test_unscaled - y_pred_test_mean
        mae_test = np.mean(np.absolute(y_pred_VS_true_test))
        rmse_test = np.sqrt(np.mean(np.square(y_pred_VS_true_test)))
        corrcoef_test = r2_score(y_test_unscaled, y_pred_test_mean)
        logging.info("For the test set:\nMAE: {0:0.4f} RMSE: {1:0.4f} R^2: {2:0.4f}\n".\
              format(mae_test, rmse_test, corrcoef_test))

        # Summarize the prediction scores
        scores_summary['train'].append([mae_train, rmse_train, corrcoef_train])
        scores_summary['valid'].append([mae_valid, rmse_valid, corrcoef_valid])
        scores_summary['test'].append([mae_test, rmse_test, corrcoef_test])
        
        # Plot the final result
        # Changed colors, scaling and sizes
        plt.figure(figsize=(12, 8))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setting plot limits
        y_true_min = min(np.min(y_train_unscaled), np.min(y_valid_unscaled), np.min(y_test_unscaled))
        y_true_max = max(np.max(y_train_unscaled), np.max(y_valid_unscaled), np.max(y_test_unscaled))
        y_pred_min = min(np.min(y_pred_train_mean), np.min(y_pred_valid_mean), np.min(y_pred_test_mean))
        y_pred_max = max(np.max(y_pred_train_mean), np.max(y_pred_valid_mean), np.max(y_pred_test_mean))
        # Expanding slightly the canvas around the data points (by 10%)
        axmin = y_true_min-0.1*(y_true_max-y_true_min)
        axmax = y_true_max+0.1*(y_true_max-y_true_min)
        aymin = y_pred_min-0.1*(y_pred_max-y_pred_min)
        aymax = y_pred_max+0.1*(y_pred_max-y_pred_min)

        plt.xlim(min(axmin, aymin), max(axmax, aymax))
        plt.ylim(min(axmin, aymin), max(axmax, aymax))
                        
        plt.errorbar(y_train_unscaled, 
                    y_pred_train_mean,
                    fmt='o',
                    label="Train",
                    elinewidth = 0, 
                    ms=5,
                    mfc='#519fc4',
                    markeredgewidth = 0,
                    alpha=0.7)
        plt.errorbar(y_valid_unscaled,
                    y_pred_valid_mean,
                    elinewidth = 0,
                    fmt='o',
                    label="Validation", 
                    ms=5, 
                    mfc='#db702e',
                    markeredgewidth = 0,
                    alpha=0.7)
        plt.errorbar(y_test_unscaled,
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
        
        if ifold == (k_fold_number-1):
            logging.info("\n*******************************")
            logging.info("***Predictions score summary***")
            logging.info("*******************************\n")
            
            scores_summary_train = np.asarray(scores_summary['train'])
            scores_summary_valid = np.asarray(scores_summary['valid'])
            scores_summary_test = np.asarray(scores_summary['test'])
            
            scores_summary_train_mean = np.mean(scores_summary_train, axis=0)
            scores_summary_valid_mean = np.mean(scores_summary_valid, axis=0)
            scores_summary_test_mean = np.mean(scores_summary_test, axis=0)
            
            scores_summary_train_std = np.std(scores_summary_train, axis=0)
            scores_summary_valid_std = np.std(scores_summary_valid, axis=0)
            scores_summary_test_std = np.std(scores_summary_test, axis=0)
            
            logging.info("For the training sets:\nMAE: {0:0.4f} +/- {1:0.4f} RMSE: {2:0.4f} +/- {3:0.4f} R^2: {4:0.4f} +/- {5:0.4f}\n".\
              format(scores_summary_train_mean[0], scores_summary_train_std[0], 
                     scores_summary_train_mean[1], scores_summary_train_std[1], 
                     scores_summary_train_mean[2], scores_summary_train_std[2]))
            logging.info("For the validation sets:\nMAE: {0:0.4f} +/- {1:0.4f} RMSE: {2:0.4f} +/- {3:0.4f} R^2: {4:0.4f} +/- {5:0.4f}\n".\
              format(scores_summary_valid_mean[0], scores_summary_valid_std[0], 
                     scores_summary_valid_mean[1], scores_summary_valid_std[1], 
                     scores_summary_valid_mean[2], scores_summary_valid_std[2]))
            logging.info("For the test sets:\nMAE: {0:0.4f} +/- {1:0.4f} RMSE: {2:0.4f} +/- {3:0.4f} R^2: {4:0.4f} +/- {5:0.4f}\n\n".\
              format(scores_summary_test_mean[0], scores_summary_test_std[0], 
                     scores_summary_test_mean[1], scores_summary_test_std[1], 
                     scores_summary_test_mean[2], scores_summary_test_std[2]))
            
            logging.info("*******************************************")
            logging.info("***SMILES_X has terminated successfully.***")
            logging.info("*******************************************")
            print("***SMILES_X has terminated successfully.***\n")
