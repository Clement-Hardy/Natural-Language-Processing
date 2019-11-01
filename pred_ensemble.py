from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GlobalAveragePooling1D
from keras.layers import GRU, CuDNNGRU, TimeDistributed, Dense, CuDNNLSTM
from keras.layers import Conv1D, Concatenate, BatchNormalization
import numpy as np

from AttentionWithContext import AttentionWithContext
from multi_head import MultiHeadAttention, Position_Embedding
from utils import bidir_lstm, bidir_gru, concatenate_embeddings, mean_pred, mean_square_error, get_train_test, get_test
from SelfAttentiveEmbeddings import SelfAttentiveEmbeddings
from kmax_pooling import KMaxPooling
from config import *
from models import get_model_1, get_model_2, get_model_3, get_model_4


"""
print the summarry of each model val loss on each target
and the ensemble val loss on this target

example:
    
******** target 0 ***********
Model 1: 0.25
Model 2: 0.35
Model 1: 0.23
Model 2: 0.27
Model 1: 0.26
Ensemble: 0.21
"""
for idx in range(len(target_idx)):
    nb_model = 1
    list_pred = []
    # begin to laod the weights and predict on the validation documents
    for i in range(len(type_loss)):
        name_doc = list_name_documents[i]
        docs_train, target_train, docs_val, target_val = get_train_test(name_doc=name_doc,
                                                                        idx_target=idx,
                                                                        config=config)
        if type_loss[i] == "mse":
            custom_loss = "mean_squared_error"
        elif type_loss[i] == "higher":
            custom_loss = mse_asymetric_higher
        elif type_loss[i] == "lower":
            custom_loss = mse_asymetric_lower
        
        if type_model[i] == 1:
            model = get_model_1(docs_train=docs_train,
                                config=config,
                                name_embeddings=list_embeddings[i],
                                custom_loss=custom_loss)
        elif type_model[i] == 2:
            model = get_model_2(docs_train=docs_train,
                                config=config, name_embeddings=list_embeddings[i],
                                custom_loss=custom_loss)
        elif type_model[i] == 3:
            model = get_model_3(docs_train=docs_train,
                                config=config,
                                name_embeddings=list_embeddings[i],
                                custom_loss=custom_loss)
        elif type_model[i] == 4:
            model = get_model_4(docs_train=docs_train,
                                config=config,
                                name_embeddings=list_embeddings[i],
                                custom_loss=custom_loss)
            
        model.load_weights(config.path_to_data + "//weight_model//" + 'target_{}__model_'.format(idx) + str(nb_model))
        preds = model.predict(docs_val)
        list_pred.append(preds)
        nb_model += 1
        
    # calculate the average prediction of the models
    # which correspond to the ensemble prediction 
    mean_pred_data = mean_pred(list_pred)
    
    print('* * * * * * *target {}'.format(idx),'* * * * * * *')
    for i in range(len(list_pred)):
        print("Model {}: ".format(i), mean_square_error(target_val, list_pred[i]))
     
    print("Ensemble : ", mean_square_error(target_val, mean_pred_data))
   

"""
Create a file of the ensemble prediction
 in the corresponding format for a submission on kaggle
"""

all_preds_han = []
for i in range(len(target_idx)):
    idx = target_idx[i]
    nb_model = 1
    list_pred = []
    # each model begin to predict on the documents test 
    for i in range(len(type_loss)):
    
        name_doc = list_name_documents[i]
        docs_val = get_test(name_doc=name_doc,
                            idx_target=idx,
                            config=config)
        
        if type_loss[i] == "mse":
            custom_loss = "mean_squared_error"
        elif type_loss[i] == "higher":
            custom_loss = mse_asymetric_higher
        elif type_loss[i] == "lower":
            custom_loss = mse_asymetric_lower

        if type_model[i] == 1:
            model = get_model_1(docs_train=docs_val,
                                config=config,
                                name_embeddings=list_embeddings[i],
                                custom_loss=custom_loss)
        elif type_model[i] == 2:
            model = get_model_2(docs_train=docs_val,
                                config=config, name_embeddings=list_embeddings[i],
                                custom_loss=custom_loss)
        elif type_model[i] == 3:
            model = get_model_3(docs_train=docs_val,
                                config=config,
                                name_embeddings=list_embeddings[i],
                                custom_loss=custom_loss)
        elif type_model[i] == 4:
            model = get_model_4(docs_train=docs_val,
                                config=config,
                                name_embeddings=list_embeddings[i],
                                custom_loss=custom_loss)
            
        model.load_weights(config.path_to_data + "//weight_model//" + 'target_{}__model_'.format(idx) + str(nb_model))
        preds = model.predict(docs_val)
        list_pred.append(preds)
        nb_model +=1
    
    # calculate the average prediction of the models
    # which correspond to the ensemble prediction 
    mean_pred_data = mean_pred(list_pred)
        
    all_preds_han.append(mean_pred_data.tolist())


# create the file for the submission on the kaggle
all_preds_han = [elt for sublist in all_preds_han for elt in sublist]
with open(config.path_to_data + 'predictions_han.txt', 'w') as file:
    file.write('id,pred\n')
    for idx,pred in enumerate(all_preds_han):
        pred = format(pred, '.7f')
        file.write(str(idx) + ',' + pred + '\n')
