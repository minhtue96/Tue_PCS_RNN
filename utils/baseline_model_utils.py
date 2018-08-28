import numpy as np


#load handcrafted features for baseline Logistic Regression models
def load_baseline_data(slide_sequences, slide_labels, correct_images, slide_id, num_features, feature=None):
    '''
    Arguments:
        slide_sequences, slide_labels: data gotten from LoadData
        correct_images: array of 21 elements, each is the correct variant of that particular slide. gotten from LoadData
        slide_id: range from 0 to 20
        num_baseline_features: can be either 1,6,7 or 8
        feature: only important if num_features=1. Choice: 'master','correct','length'
    Return: X and y
    '''
    NUM_SUBJECTS = slide_labels.shape[1]
    X = np.empty((NUM_SUBJECTS, num_features),dtype=float)
    y = slide_labels[slide_id]
    for subject_id in range(NUM_SUBJECTS):
        if len(slide_sequences[slide_id][subject_id]) != 0:
            count_fix = np.bincount(slide_sequences[slide_id][subject_id], minlength=8)
            length = count_fix.sum()
        else:
            count_fix = np.zeros(8)
            length = 0
        
        if num_features == 1:
            if feature == 'master':
                X[subject_id,0] = count_fix[1]
            elif feature == 'correct':
                corr_img = correct_images[slide_id]
                X[subject_id,0] = count_fix[corr_img]
            elif feature == 'length':
                X[subject_id,0] = length
            else:
                print('Invalid argument: feature')
                return X,y
        elif num_features == 6:
            X[subject_id][:6] = count_fix[2:8]
        elif num_features == 7:
            X[subject_id][:7] = count_fix[1:8]
        elif num_features == 8:
            X[subject_id][:7] = count_fix[1:8]
            X[subject_id][7] = length
        else:
            print('Invalid argument: num_features')
            return X,y
    return X,y


