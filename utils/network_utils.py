import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed) 


from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU
from keras.callbacks import EarlyStopping
from keras import optimizers

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold



def make_model(max_sequence_len, outputs):
    model = Sequential()	
    model.add(Embedding(8,128, input_length = max_sequence_len))
    model.add(GRU(32, dropout=0.5, recurrent_dropout=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(outputs, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0003), metrics=['accuracy'])
    return model
'''
def make_model(max_sequence_len):
    model = Sequential()	
    model.add(Embedding(8,256, input_length = max_sequence_len))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0003), metrics=['accuracy'])
    return model
'''
