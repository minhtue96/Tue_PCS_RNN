import numpy as np
seed = 7
np.random.seed(seed)

NUM_SLIDES = 21

from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences



class AblationSlide:    
    def __init__(self, slide_label, model, max_sequence_length, ablated_sequence):
        '''
        Arguments:
            slide_label: Get from LoadData class. Selections of original sequences for this slide. Shape: (NUM_SUBJECTS)
            model: k-fold models for this slide. Shape: (k)
            max_sequence_length: int. Max sequence length for this slide. Get from Network class
            ablated_sequence: slide_sequence for this slide after modified. List of NUM_SUBJECTS sequences (before pad)
        '''
        k = model.shape[0]
        NUM_SUBJECTS = slide_labels.shape[0]
        stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
        X = pad_sequences(ablated_sequence, maxlen=max_sequence_length)
        y = slide_label
        
        fold = 0
        for train,test in stratified_kfold.split(X,y):
            y_prob = model[fold].predict(X[test]).flatten()
            y_pred = model[fold].predict_classes(X[test], verbose=0).flatten()
            abla_conf[slide_id,fold] = confusion_matrix(y[test],y_pred)
            abla_acc[slide_id,fold] = accuracy_score(y[test],y_pred)
            abla_auc[slide_id,fold] = roc_auc_score(y[test],y_prob)
            abla_pred[slide_id,test] = y_pred
            abla_prob[slide_id,test] = y_prob
            fold += 1
                
    
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
