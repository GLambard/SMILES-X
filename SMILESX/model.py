from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Bidirectional, TimeDistributed, LSTM
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Layer

from tensorflow.keras import backend as K
import tensorflow as tf

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
    def __init__(self, return_probabilities = False, seed = None, **kwargs):
        self.return_probabilities = return_probabilities
        self.seed = seed
        super(AttentionM, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS,)
        self.W = self.add_weight(name="W_{:s}".format(self.name), 
                                 shape=(input_shape[-1], 1),
                                 initializer=tf.keras.initializers.GlorotNormal(seed=self.seed))
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
class LSTMAttModel:
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
    def create(inputtokens, vocabsize, 
               lstmunits=16, denseunits=16, embedding=32, 
               return_proba = False, 
               seed = None):

        input_ = Input(shape=(inputtokens,), dtype='int32')

        # Embedding layer
        net = Embedding(input_dim=vocabsize, 
                        output_dim=embedding, 
                        input_length=inputtokens, 
                        embeddings_initializer=tf.keras.initializers.he_uniform(seed=seed))(input_)

        # Bidirectional LSTM layer
        net = Bidirectional(LSTM(lstmunits, 
                                 return_sequences=True, 
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed), 
                                 recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=seed)))(net)
        net = TimeDistributed(Dense(denseunits, 
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))(net)
        net = AttentionM(return_probabilities=return_proba, seed=seed)(net)

        # Output layer
        net = Dense(1, activation="linear", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(net)
        model = Model(inputs=input_, outputs=net)

        return model
##
