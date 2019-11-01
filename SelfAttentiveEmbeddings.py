import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints


def dot_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class SelfAttentiveEmbeddings(Layer):
    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, hidden_dim=350, b_constraint=None,
                 bias=False, n_hops=3, penalty=0.1, **kwargs):
        self.supports_masking = True
        self.hidden_dim = hidden_dim
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.n_hops = n_hops
        self.P = penalty

        self.bias = bias
        super(SelfAttentiveEmbeddings, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((self.hidden_dim, input_shape[-1], ),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((self.n_hops, self.hidden_dim, ),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(SelfAttentiveEmbeddings, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        
        if self.bias:
            uit += self.b
        
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        
        softmax1 = K.softmax(ait, axis=0)
        A = K.permute_dimensions(softmax1, (0, 2, 1))

        M = K.batch_dot(A, x, axes=[2, 1])

        reshape = K.batch_flatten(M)
        eye = K.eye(self.n_hops)
        prod = K.batch_dot(softmax1, A, axes=[1, 2])
        
        #self.add_loss(self.P * K.sqrt(K.sum(K.square(prod - eye))))
        
        return reshape
    
    def compute_output_shape(self, input_shape):
            return (input_shape[0], self.n_hops * input_shape[-1],)
