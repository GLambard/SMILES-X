from keras.models import Model

from keras.layers import Input, Dense
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers import CuDNNLSTM, TimeDistributed

from keras.engine.topology import Layer

from keras.utils import multi_gpu_model

from keras import backend as K
# import tensorflow as tf

# #from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# K.set_session(sess)  # set this TensorFlow session as the default session for Keras

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