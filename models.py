# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:52:55 2019

@author: Hard Med lenovo
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GlobalAveragePooling1D
from keras.layers import GRU, CuDNNGRU, TimeDistributed, Dense, CuDNNLSTM
from keras.layers import Conv1D, Concatenate, BatchNormalization
import numpy as np

from AttentionWithContext import AttentionWithContext
from multi_head import MultiHeadAttention, Position_Embedding
from utils import bidir_lstm, bidir_gru, concatenate_embeddings
from SelfAttentiveEmbeddings import SelfAttentiveEmbeddings
from kmax_pooling import KMaxPooling


def get_model_1(docs_train, config, name_embeddings=None, custom_loss="mean_squared_error"):
    """
    This model correspond to the baseline model
    
    Args:
        docs_train: numpy array
                    the document to train the model
        config: object
                contain the necessary parametes to build the model (n_hidden, is_GPU, type of optimizer....)
        name_embeddings: str
                        optionnal, a name of file containing a embedding matrix to concatenate to the default embeddings
        custom_loss: str
                    optionnal, name of the loss of the model
                    default, mean_square error, some
                    some loss are available in the file loss
    """

    # concantenate embeddings if an embeddings file name is pass in argument
    if name_embeddings is not None:
        embeddings1 = np.load(config.path_to_data + 'embeddings.npy')
        embeddings2 = np.load(config.path_to_data + name_embeddings)
        embeddings = concatenate_embeddings(embeddings1, embeddings2)
    else:
        embeddings = np.load(config.path_to_data + 'embeddings.npy')
        
    sent_ints = Input(shape=(docs_train.shape[2],))

    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=docs_train.shape[2],
                        trainable=False,
                        )(sent_ints)
    sent_wa = bidir_gru(sent_wv,config.n_units, config.is_GPU)

    sent_att_vec = AttentionWithContext()(sent_wa)
    sent_encoder = Model(sent_ints,sent_att_vec)

    doc_ints = Input(shape=(docs_train.shape[1],docs_train.shape[2],))
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr, config.n_units, config.is_GPU)
    doc_att_vec = AttentionWithContext()(doc_sa)
    doc_att_vec_dr = Dropout(config.drop_rate)(doc_att_vec)

    preds = Dense(units=1, activation=None)(doc_att_vec_dr)
    model = Model(doc_ints, preds)

    model.compile(loss=custom_loss,
                  optimizer=config.my_optimizer)

    print('model compiled')
    return model


def get_model_2(docs_train, config, name_embeddings=None, custom_loss="mean_squared_error"):
    """
    This model correspond to the multi head model
    
    Args:
        docs_train: numpy array
                    the document to train the model
        config: object
                contain the necessary parametes to build the model (n_hidden, is_GPU, type of optimizer....)
        name_embeddings: str
                        optionnal, a name of file containing a embedding matrix to concatenate to the default embeddings
        custom_loss: str
                    optionnal, name of the loss of the model
                    default, mean_square error, some
                    some loss are available in the file loss
    """
    
    if name_embeddings is not None:
        embeddings1 = np.load(config.path_to_data + 'embeddings.npy')
        embeddings2 = np.load(config.path_to_data + name_embeddings)
        embeddings = concatenate_embeddings(embeddings1, embeddings2)
    else:
        embeddings = np.load(config.path_to_data + 'embeddings.npy')
    
    input_model_sentence = Input(shape=(docs_train.shape[2],))
    
    x = Embedding(input_dim=embeddings.shape[0],
                  output_dim=embeddings.shape[1],
                  weights=[embeddings],
                  input_length=docs_train.shape[2],
                  trainable=False,
                  )(input_model_sentence)

    x = Dropout(config.drop_rate)(x)
    x = bidir_lstm(x, config.n_units, config.is_GPU)

    x = MultiHeadAttention(nb_head=config.n_head,
                           size_per_head=config.n_hidden_attention)([x, x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(config.drop_rate)(x)
    output_model_sentence = Dense(100)(x)
    
    sent_encoder = Model(input_model_sentence, output_model_sentence)

    input_model_doc = Input(shape=(docs_train.shape[1], docs_train.shape[2],))
    x1 = TimeDistributed(sent_encoder)(input_model_doc)

    x1 = bidir_lstm(x1, config.n_units, config.is_GPU)

    x1 = AttentionWithContext(return_coefficients=False)(x1)
    
    x1 = Dropout(config.drop_rate)(x1)

    x1 = Dense(100)(x1)
    output_model_doc = Dense(units=1)(x1)
    
    model = Model(input_model_doc, output_model_doc)

    model.compile(loss=custom_loss,
                  optimizer=config.my_optimizer,
                  metrics=['mae'])

    print('model compiled')
    
    return model


def get_model_3(docs_train, config, name_embeddings=None, custom_loss="mean_squared_error"):
    """
    This model correspond to the SelfAttentive Model
    
    Args:
        docs_train: numpy array
                    the document to train the model
        config: object
                contain the necessary parametes to build the model (n_hidden, is_GPU, type of optimizer....)
        name_embeddings: str
                        optionnal, a name of file containing a embedding matrix to concatenate to the default embeddings
        custom_loss: str
                    optionnal, name of the loss of the model
                    default, mean_square error, some
                    some loss are available in the file loss
    """
    
    if name_embeddings is not None:
        embeddings1 = np.load(config.path_to_data + 'embeddings.npy')
        embeddings2 = np.load(config.path_to_data + name_embeddings)
        embeddings = concatenate_embeddings(embeddings1, embeddings2)
    else:
        embeddings = np.load(config.path_to_data + 'embeddings.npy')
        
    input_model_sentence = Input(shape=(docs_train.shape[2],))

    x = Embedding(input_dim=embeddings.shape[0],
                  output_dim=embeddings.shape[1],
                  weights=[embeddings],
                  input_length=docs_train.shape[2],
                  trainable=False,
                  )(input_model_sentence)

    x = Dropout(config.drop_rate)(x)

    x = bidir_lstm(x, config.n_units, config.is_GPU)
    x = SelfAttentiveEmbeddings()(x)

    output_model_sentence = Dropout(config.drop_rate)(x)

    sent_encoder = Model(input_model_sentence, output_model_sentence)
    
    input_model_doc = Input(shape=(docs_train.shape[1], docs_train.shape[2],))
    x1 = TimeDistributed(sent_encoder)(input_model_doc)

    x1 = bidir_lstm(x1, config.n_units, config.is_GPU)
    x1 = AttentionWithContext(return_coefficients=False)(x1)

    x1 = Dropout(config.drop_rate)(x1)

    x1 = Dense(100)(x1)
    output_model_doc = Dense(units=1)(x1)
    
    model = Model(input_model_doc, output_model_doc)

    model.compile(loss=custom_loss,
                  optimizer=config.my_optimizer,
                  metrics=['mae'])
    
    return model

    
def get_model_4(docs_train, config, name_embeddings=None, custom_loss="mean_squared_error"):
    """
    This model correspond to the Hierarchical convolutionnal attention network
    
    Args:
        docs_train: numpy array
                    the document to train the model
        config: object
                contain the necessary parametes to build the model (n_hidden, is_GPU, type of optimizer....)
        name_embeddings: str
                        optionnal, a name of file containing a embedding matrix to concatenate to the default embeddings
        custom_loss: str
                    optionnal, name of the loss of the model
                    default, mean_square error, some
                    some loss are available in the file loss
    """
    
    if name_embeddings is not None:
        embeddings1 = np.load(config.path_to_data + 'embeddings.npy')
        embeddings2 = np.load(config.path_to_data + name_embeddings)
        embeddings = concatenate_embeddings(embeddings1, embeddings2)
    else:
        embeddings = np.load(config.path_to_data + 'embeddings.npy')
    input_model_sentence = Input(shape=(docs_train.shape[2],))

    x = Embedding(input_dim=embeddings.shape[0],
                  output_dim=embeddings.shape[1],
                  weights=[embeddings],
                  input_length=docs_train.shape[2],
                  trainable=False,
                  )(input_model_sentence)
    
    filter_sizes = [3, 4, 5]
    convs = []
    for filter_size in filter_sizes:                        
        conv = Conv1D(
                    filters=64,
                    kernel_size=filter_size, 
                    padding="same", 
                    )(x)                
        batch_normalization = BatchNormalization()(conv)          
        pool = KMaxPooling(k=5, axis=1)(batch_normalization) 
        convs.append(pool)

    x = Concatenate(axis=1)(convs)

    x = Dropout(config.drop_rate)(x)
    x = bidir_lstm(x,config.n_units, config.is_GPU)
    x = AttentionWithContext(return_coefficients=False)(x)
    output_model_sentence = Dropout(config.drop_rate)(x)                      
    sent_encoder = Model(input_model_sentence,output_model_sentence)

    input_model_doc = Input(shape=(docs_train.shape[1], docs_train.shape[2],))
    x1 = TimeDistributed(sent_encoder)(input_model_doc)
    x1 = bidir_lstm(x1, config.n_units, config.is_GPU)
    x1 = AttentionWithContext(return_coefficients=False)(x1)
    x1 = Dropout(config.drop_rate)(x1)
    x1 = Dense(100)(x1)
    
    output_model_doc = Dense(units=1)(x1)
    model = Model(input_model_doc, output_model_doc)
    
    model.compile(loss=custom_loss,
                  optimizer=config.my_optimizer,
                  metrics=['mae'])

    print('model compiled')
    
    return model
