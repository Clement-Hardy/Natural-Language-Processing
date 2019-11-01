import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Bidirectional, CuDNNLSTM, CuDNNGRU, GRU, LSTM



def bidir_gru(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh', 
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)
    
    
def bidir_lstm(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with LSTM units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNLSTM(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(LSTM(units=n_units,
                                 activation='tanh', 
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)
    
    

def get_train_test(name_doc, idx_target, config):
    """
    load the documents and the objective (train and validation) for a specific target
    
    Args:
        name_doc: str
                  name of the document to load
        idx_target: int
                    the number of the target
                    should be 0,1,2,3
        config: object
                a class containing some information of the models
                , path of the data,...
                
    Return:
            docs_train: numpy array
                        the documents to train the model
            target_train: numpy array
                          the objective of the model
            docs_val: numpy array
                        the documents to validate the model
            target_val: numpy array
                          the objective to validate the model
    
    """
    
    docs = np.load(config.path_to_data + name_doc)
    
    with open(config.path_to_data + 'train_idxs.txt', 'r') as file:
        train_idxs = file.read().splitlines()
    
    train_idxs = [int(elt) for elt in train_idxs]

    np.random.seed(12219)
    
    idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
    idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)
    
    train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
    val_idxs = [train_idxs[elt] for elt in idxs_select_val]

    docs_train = docs[train_idxs_new,:,:]
    docs_val = docs[val_idxs,:,:]

    with open(config.path_to_data + 'targets/train/target_' + str(idx_target) + '.txt', 'r') as file:
        target = file.read().splitlines()
    
    target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
    target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

    
    return docs_train, target_train, docs_val, target_val


def get_test(name_doc, idx_target, config):
     """
    load the test documents for a specific target (to do a prediction for kaggle)
    
    Args:
        name_doc: str
                  name of the document to load
        idx_target: int
                    the number of the target
                    should be 0,1,2,3
        config: object
                a class containing some information of the models
                , path of the data,...
                
    Return:
            docs_test: numpy array
                       the documents to predict the values

    
    """
    docs = np.load(config.path_to_data + name_doc)
    
    with open(config.path_to_data + 'test_idxs.txt', 'r') as file:
        test_idxs = file.read().splitlines()
    
    test_idxs = [int(elt) for elt in test_idxs]
    docs_test = docs[test_idxs,:,:]
    
    return docs_test


def concatenate_embeddings(embeddings1, embeddings2):
    """
    embeddings1 and embeddings2 should have the same number of column
    Concatenate two embeddings matrix, the resulting matrix have the 
    same number of row, the number of column is the number of column of embeddings1 + the number of column of embeddings2   
    """
    return np.concatenate((embeddings1, embeddings2), axis=1)

def mean_pred(list_pred):
    
    """
    make the prediction of the ensemble model
    
    Args:
        list_pred: list
                    a list containing the prediction (numpy array) of each model of the model
                    each predictions should have the same dimensions
    """    
    preds = np.zeros((len(list_pred[0]), len(list_pred)))
    for i in range(len(list_pred)):
        if list_pred[i].ndim==2:
            preds[:,i] = list_pred[i][:,0]
        else:
            preds[:,i] = list_pred[i]
    return np.mean(preds, axis=1)


        
