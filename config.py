# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:43:36 2019

@author: Hard Med lenovo
"""
from keras import optimizers


class config:
    # class containing the parameter of the models
    is_GPU = True
    save_weights = True
    save_history = True
    
    path_root = "C://Users//Hard Med lenovo//Desktop//MATH M2 Data science//Advanced text and graph data//kaggle"
    path_to_code = path_root + '//code//'
    path_to_data = path_root + '//data//'
    
    n_units = 100
    drop_rate = 0.01
    batch_size = 32
    nb_epochs = 20
    my_optimizer = optimizers.adam(lr=1e-3)
    my_patience = 40
    n_head = 2
    n_hops = 10
    n_hidden_attention = 64
    

# a list containing the name of the documents created by node2vec
# and choosen for the model, each model have one document
list_name_documents = ["documents_4_4.npy", "documents10_5_10_6_4.npy", "documents_025_4.npy", "documents_1_1.npy", "documents_1_4.npy"]

# the two following list select what the model will do, train or not, load pretrained weight (train in the pas)
train_mode = [True, True, True, True, True, True]
load_mode = [False, False, False, False, False, False]

# the two following list select the type of model (corresponding to the number of the function in file models)
# and the type of loss (two different loss are availabel in file loss)
type_model = [1, 1, 1, 1, 1, 1]
type_loss = ["mse", "mse", "mse", "mse", "mse"]

# name of embedding which will be concatenate with the default embeddings
# by default None embeddings will concatenate
list_embeddings = [None, None, None, None, None, None]
target_idx =[0, 1, 2, 3]