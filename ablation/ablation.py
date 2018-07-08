import numpy as np
seed = 7
np.random.seed(seed)

NUM_SLIDES = 21

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences



class Ablation:    
    def __init__(self, slide_labels, model, max_sequence_lengths, ablated_sequences):
        '''
        Arguments:
            slide_labels: Get from LoadData class. Selections of original sequences. Shape: (NUM_SLIDES, NUM_SUBJECTS)
            model: models for 21 slides, each slide has k folds
            max_sequence_lengths: list of 21 elements, each has the maximum sequence length in a slide. Get from Network class
            ablated_sequences: slide_sequences after modified. List of 21 elements, each is list of NUM_SUBJECTS sequences
        '''
        k = model.shape[1]
        NUM_SUBJECTS = slide_labels.shape[1]
        stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        abla_acc,abla_auc,abla_conf = np.empty((NUM_SLIDES,k)),np.empty((NUM_SLIDES,k)),np.empty((NUM_SLIDES,k,2,2))
        abla_pred,abla_prob = np.empty((NUM_SLIDES,NUM_SUBJECTS)),np.empty((NUM_SLIDES,NUM_SUBJECTS))
        
        for slide_id in range(NUM_SLIDES):
            X = pad_sequences(ablated_sequences[slide_id], maxlen=max_sequence_lengths[slide_id])
            y = slide_labels[slide_id]
            
            fold = 0
            for train,test in stratified_kfold.split(X,y):
                y_prob = model[slide_id,fold].predict(X[test]).flatten()
                y_pred = model[slide_id,fold].predict_classes(X[test], verbose=0).flatten()
                abla_conf[slide_id,fold] = confusion_matrix(y[test],y_pred)
                abla_acc[slide_id,fold] = accuracy_score(y[test],y_pred)
                abla_auc[slide_id,fold] = roc_auc_score(y[test],y_prob)
                abla_pred[slide_id,test] = y_pred
                abla_prob[slide_id,test] = y_prob
                fold += 1
                
        self.accuracy = abla_acc
        self.confusion_matrix = abla_conf.astype(int)
        self.auc_score = abla_auc
        self.y_predict = abla_pred
        self.y_probability = abla_prob
    
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
        return self.confusion_matrix.sum(axis=1)
    
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