import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed) 

NUM_SLIDES = 21
epochs = 200
patience = 15

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.models import load_model
from keras.utils import to_categorical

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from utils.network_utils import make_model ##########
import pickle


class Network:  #after making a Network object, the first thing to do is either train() if first time training,
                # or load() if already trained the network before and already had the saved files for models, accuracies, etc.
    def __init__(self):
        self.k = 5   #5 folds
        self.model = [[] for slide_id in range(NUM_SLIDES)]
    
    def train_and_save(self, slide_sequences, slide_labels, model_type_name, should_save_auc): #should saved() right after train()

	if model_type_name == "SELECTION":
		outputs = 6
        else:
		outputs = 2
	# make directories:
	if not os.path.exists('network/models{}'.format(model_type_name)):
		os.mkdir('network/models{}'.format(model_type_name))
		os.mkdir('network/histories{}'.format(model_type_name))
		os.mkdir('network/scores{}'.format(model_type_name))

        NUM_SUBJECTS = slide_labels.shape[1]
        max_sequence_lengths = []
        slide_sequences_pad = []
        for slide_id in range(NUM_SLIDES):
            slide_sequences_pad.append(pad_sequences(slide_sequences[slide_id]))
            max_sequence_lengths.append(len(slide_sequences_pad[slide_id][0]))
        
        k = self.k
        stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
        accu,conf,aucroc,mod,hist,pred,prob = [],[],[],[],[],[],[]
        for slide_id in range(NUM_SLIDES):
            X = slide_sequences_pad[slide_id]
            y = slide_labels[slide_id]
	    if model_type_name == "SELECTION":
		y = y - 1
	    y_cat = np.zeros(shape=(y.shape[0], outputs))
	    for i, y_i in enumerate(y):
		y_cat[i, y_i] = 1

            cvscores = []
            conf_matrices = []
            auc_scores = []
            models = []
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
            histories = []
            fold = 0
            

            y_probs = np.empty(shape=(NUM_SUBJECTS, outputs))
            y_preds = np.empty(NUM_SUBJECTS)
            max_sequence_len = max_sequence_lengths[slide_id]
            
            for train, test in stratified_kfold.split(X, y):
                class_weight = compute_class_weight(class_weight='balanced', 
			classes=np.unique(y[train]), y=y[train])
                class_weight = dict(zip(np.unique(y[train]), class_weight))

                model = make_model(max_sequence_len, outputs)

                history = model.fit(X[train], y_cat[train], epochs=epochs, batch_size=64, 
			class_weight=class_weight, verbose=0, 
			validation_data=(X[test], y_cat[test]),
			callbacks=[early_stopping])
                histories.append(history)
                models.append(model)
                
                y_pred = model.predict_classes(X[test], verbose=0).flatten()
                y_preds[test] = y_pred
                conf_matrix = confusion_matrix(y[test], y_pred)
		if model_type_name == "SELECTION" and conf_matrix.shape != (6, 6):
			conf_dict = {}
			for i in range(0, conf_matrix.shape[0]):
				for j in range(0, conf_matrix.shape[1]):
					i_val = np.unique(np.append(y[test], y_pred))[i]
					j_val = np.unique(np.append(y[test], y_pred))[j]
					conf_dict[i_val, j_val] = conf_matrix[i, j]
			
			new_conf_mat = np.zeros(shape=(6, 6))
			for i in range(0, 6):
				for j in range(0, 6):

					if (i, j) not in conf_dict:
						new_conf_mat[i, j] = 0
					else:

						new_conf_mat[i, j] = conf_dict[(i, j)]
			conf_matrix = new_conf_mat
                conf_matrices.append(conf_matrix)
                acc = accuracy_score(y[test], y_pred)
                cvscores.append(acc)
                y_prob = model.predict(X[test])
                y_probs[test] = y_prob
		if should_save_auc:
                	auc = roc_auc_score(y[test], y_prob)
                	auc_scores.append(auc)
                	print('\tAUC:', auc)
                
                print('\tAccuracy:', acc)
                print('\tConfusion matrix:\n', conf_matrix)
                print('\n\n')

                #save model
                print('Saving Slide', slide_id+1, 'Fold', fold+1)
                model.save('network/models{}/model{}_{}.h5'.format(
			model_type_name, slide_id+1, fold+1))

                with open('network/histories{}/history{}_{}.pickle'.format(
			model_type_name, slide_id+1, fold+1), 'wb') as pickle_file:
                    pickle.dump(history.history, pickle_file)
                fold+=1
            
            cvscores = np.array(cvscores)

            conf_matrices = np.mean(np.dstack(conf_matrices), axis=2)
	    if should_save_auc:
            	auc_scores = np.array(auc_scores)
            
            accu.append(cvscores)
            conf.append(conf_matrices)
            if should_save_auc:
		aucroc.append(auc_scores)
            pred.append(y_preds)
            prob.append(y_probs)
            hist.append(histories)
            mod.append(models)
        accu = np.array(accu)

	conf = np.array(conf)
        if should_save_auc:
        	aucroc = np.array(aucroc)
        pred = np.array(pred)
        prob = np.array(prob)
        hist = np.array(hist)
        mod = np.array(mod)
        
        #save other variables
        with open('network/scores{}/accuracies.pickle'.format(model_type_name), 'wb') as pickle_file:
            pickle.dump(accu, pickle_file)
        if should_save_auc:
		filename = 'network/scores{}/auc_scores.pickle'.format(model_type_name)
        	with open(filename, 'wb') as pickle_file:
			pickle.dump(aucroc, pickle_file)
	filename2 = 'network/scores{}/confusion_matrices.pickle'.format(model_type_name)
        with open(filename2, 'wb') as pickle_file:
          	pickle.dump(conf, pickle_file)
        with open('network/scores{}/y_predicts.pickle'.format(model_type_name), 'wb') as pickle_file:
            pickle.dump(pred, pickle_file)
        with open('network/scores{}/y_probabilities.pickle'.format(model_type_name), 'wb') as pickle_file:
            pickle.dump(prob, pickle_file)

        self.accuracy = accu
        if should_save_auc:
        	self.confusion_matrix = conf
        	self.auc_score = aucroc
        self.y_predict = pred
        self.y_probability = prob
        self.history = hist
        self.model = mod

    def load_models(self, model_type_name):
        for slide_id in range(NUM_SLIDES):
            for fold in range(self.k):
                #load_models
                print('Load model for slide', slide_id+1, 'fold', fold+1)
		filename = 'network/models{}/model{}_{}.h5'.format(
			model_type_name, slide_id+1, fold+1)
                self.model[slide_id].append(load_model(filename))
        self.model = np.array(self.model)
    
    def load_histories(self, model_type_name):
        self.history = []
        for slide_id in range(NUM_SLIDES):
            self.history.append([])
            for fold in range(self.k):
                #load histories
		filename = 'network/histories{}/history{}_{}.pickle'.format(model_type_name, 
			slide_id+1, fold+1)
                self.history[slide_id].append(pickle.load(open(filename, 'rb')))
        self.history = np.array(self.history)
        
    def load_results(self, model_type_name):
        #load accuracy, auc, conf matrix, y_predict, y_proba
	filename = 'network/scores{}'.format(model_type_name) 
        self.accuracy = pickle.load(open(filename+'/accuracies.pickle', 'rb'))
        #self.auc_score = pickle.load(open(filename+'/auc_scores.pickle', 'rb'))
        self.confusion_matrix = pickle.load(open(filename+'/confusion_matrices.pickle', 'rb'))
        self.y_predict = pickle.load(open(filename+'/y_predicts.pickle', 'rb'))
        self.y_probability = pickle.load(open(filename+'/y_probabilities.pickle', 'rb'))
            
                
    def get_accuracies(self):
        '''
        return: list of 21 elements, each is the average accuracy of the 5 folds in a paricular slide
        '''
        return self.accuracy.mean(axis=1)
        
    def get_auc_scores(self):
        '''
        return: list of 21 elements, each is the average auc score of the 5 folds in a paricular slide
        '''
        return self.auc_score.mean(axis=1)
    
    def get_confusion_matrices(self):
        '''
        return: list of 21 elements, each is the total confusion matrices of the 5 folds in a paricular slide
        '''
	import pdb; pdb.set_trace()
        return self.confusion_matrix
        
    def get_y_predict(self):
        '''
        return: list of 21 elements, each is a list of predicted class for each subject. Shape: NUM_SLIDES x NUM_SUBJECTS
        '''
        return self.y_predict
    
    def get_y_probability(self):
        '''
        return: list of 21 elements, each is a list of probability of correctness for each subject. Shape: NUM_SLIDES x NUM_SUBJECTS
        '''
        return self.y_probability
    
    def get_histories(self):
        '''
        return: list of 21 elements, each is a list of k Keras history objects (for the k fold). Shape: NUM_SLIDES x k
        '''
        return self.history
    
    def get_models(self):
        '''
        return: list of 21 elements, each is a list of k models (for the k fold). Shape: NUM_SLIDES x k
        '''
        return self.model


    #plot validation loss in 3 figures, each have 7 slides
    def plot_val_loss(self):
        num_dev = 3 #number of sub graphs for validation loss e.g. For num_dev = 3, we'll draw 3 graphs, each with 7 slides
        line_style_arr = ['-','--',':','o-','s-','^-','x-']
        
        hist = self.history
        
        for dev_id in range(num_dev):
            plt.figure()
            plt.xlabel('Epochs')
            plt.ylabel('Validation loss')
            legend_names = []
            for slide_id in range(int(dev_id*NUM_SLIDES/num_dev),int((dev_id+1)*NUM_SLIDES/num_dev)):
                val_loss_arr = np.ma.empty((self.k,epochs))
                val_loss_arr.mask = True
                for fold in range(self.k):
                    val_loss_arr[fold,:len(hist[slide_id,fold]['val_loss'])] = hist[slide_id,fold]['val_loss']
                val_loss_arr = val_loss_arr.mean(axis=0)
                plt.plot(np.arange(epochs, step=5), val_loss_arr[::5], line_style_arr[int(slide_id-dev_id*NUM_SLIDES/num_dev)])
                legend_names.append('Slide '+str(slide_id+1))
            plt.legend(legend_names, borderpad=3, labelspacing=2)
        plt.show(block=True)
