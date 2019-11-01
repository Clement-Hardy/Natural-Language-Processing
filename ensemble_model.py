import sys
import numpy as np
import json

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional
from keras.layers import GRU, CuDNNGRU, TimeDistributed, Dense, CuDNNLSTM


from utils import plot_pred, get_train_test
from models import get_model_1, get_model_2, get_model_3, get_model_4
from multi_head import MultiHeadAttention, Position_Embedding
from loss import mse_asymetric_higher, mse_asymetric_lower
from config import *


sys.path.insert(0, config.path_to_code)

# train the models for each target
for idx in range(len(type_loss)):
    
    nb_model = 1  # begin with the first model of the ensemble
    list_models = []
    list_docs_val = []
    
    # train each model of the ensemble
    for i in range(len(type_loss)):
    
        # get the docs and objective of the corresponding target
        name_doc = list_name_documents[i]
        docs_train, target_train, docs_val, target_val = get_train_test(name_doc=name_doc,
                                                                    idx_target=idx,
                                                                    config=config)
        list_docs_val.append(docs_val)
    
        # take the type of loss chosen for the model
        if type_loss[i] == "mse":
            custom_loss = "mean_squared_error"
        elif type_loss[i] == "higher":
            custom_loss = mse_asymetric_higher
        elif type_loss[i] == "lower":
            custom_loss = mse_asymetric_lower

        # take (create) the type of model chosen
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
        
        if load_mode[i]:
            model.load_weights(config.path_to_data + "//weight_model//" + 'target_{}__model_'.format(idx) + str(nb_model))
        list_models.append(model)

        if train_mode[i]:
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=config.my_patience,
                                           mode='min')

            checkpointer = ModelCheckpoint(filepath=config.path_to_data + "//weight_model//" + 'target_{}__model_'.format(idx) + str(nb_model), 
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True)

            if config.save_weights:
                my_callbacks = [early_stopping,checkpointer]
            else:
                my_callbacks = [early_stopping]

            model.fit(docs_train, 
                      target_train,
                      batch_size = config.batch_size,
                      epochs = config.nb_epochs,
                      validation_data = (docs_val, target_val),
                      callbacks=my_callbacks)

            hist = model.history.history

            if config.save_history:
                with open(config.path_to_data + "//history_model//" + 'target_{}_history__model_'.format(idx) + str(nb_model) + '.json', 'w') as file:
                        json.dump(hist, file, sort_keys=False, indent=4)
        # next model
        nb_model += 1
