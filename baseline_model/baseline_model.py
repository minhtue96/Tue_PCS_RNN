import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from utils.baseline_model_utils import load_baseline_data

NUM_SLIDES = 21
seed = 7
np.random.seed(seed)


class BaselineModel:
    k = 5   #5 fold

    def train(self, slide_sequences, slide_labels, correct_images, num_features, feature):
        k = self.k
        NUM_SUBJECTS = slide_labels.shape[1]
        stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        logreg_mod = []
        logreg_pred,logreg_prob = np.empty((NUM_SLIDES,NUM_SUBJECTS)),np.empty((NUM_SLIDES,NUM_SUBJECTS))
        #logreg_acc,logreg_auc,logreg_conf = np.empty((NUM_SLIDES,k)),np.empty((NUM_SLIDES,k)),np.empty((NUM_SLIDES,k,2,2))     #uncomment for correct/incorrect model
        logreg_acc = np.empty((NUM_SLIDES,k))   #uncomment for selection model
        for slide_id in range(NUM_SLIDES):
            X,y = load_baseline_data(slide_sequences, slide_labels, correct_images, slide_id, num_features=num_features, feature=feature)
            logreg_mod.append([])
            fold = 0
            for train,test in stratified_kfold.split(X,y):
                log_reg = LogisticRegression(solver="liblinear", class_weight='balanced', random_state=seed)
                log_reg.fit(X[train], y[train])
                y_pred = log_reg.predict(X[test])
                y_prob = log_reg.predict_proba(X[test])[:,1]
                #logreg_conf[slide_id,fold] = confusion_matrix(y[test],y_pred)   #uncomment for correct/incorrect model
                logreg_acc[slide_id,fold] = accuracy_score(y[test],y_pred)
                #logreg_auc[slide_id,fold] = roc_auc_score(y[test],y_prob)       #uncomment for correct/incorrect model
                logreg_pred[slide_id,test] = y_pred
                logreg_prob[slide_id,test] = y_prob
                logreg_mod[slide_id].append(log_reg)
                fold+=1
        
        self.model = np.array(logreg_mod)
        self.accuracy = logreg_acc
        #self.confusion_matrix = logreg_conf     #uncomment for correct/incorrect model
        #self.auc_score = logreg_auc     #uncomment for correct/incorrect model
        self.y_predict = logreg_pred
        self.y_probability = logreg_prob
    
    
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
    
    def get_models(self):
        '''
        return: list of 21 elements, each is a list of k models (for the k fold). Shape: NUM_SLIDES x k
        '''
        return self.model

        
