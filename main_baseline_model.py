from load_data.load_data import LoadData
from baseline_model.baseline_model import BaselineModel
import numpy as np
import tensorflow as tf
import argparse

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed) 

NUM_SLIDES = 21

if __name__ == '__main__':
    description = 'Get performance of baseline logistic regression models'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    
    num_features_choices = ['1c','1m','1l','6','7','8']
    help_num_features = 'Input number of handcrafted features.\n'+\
        '1c: 1 feature - number of fixations on correct variant\n'+\
        '1m: 1 feature - number of fixations on master image\n'+\
        '1l: 1 feature - sequence length\n'+\
        '6: 6 features - number of fixations on 6 variants (without master image)\n'+\
        '7: 7 features - number of fixations on all 7 variants (including master image)\n'+\
        '8: 8 features - number of fixations on all 7 variants and sequence length\n'
    
    parser.add_argument('num_features', help=help_num_features, choices=num_features_choices)
    
    parser.add_argument('--conf', help='Show confusion matrices', action="store_true")
    parser.add_argument('--acc', help='Show accuracies', action="store_true")
    parser.add_argument('--auc', help='Show auc_scores', action="store_true")
    
    args = parser.parse_args()
    num_features = args.num_features
    feature = None
    if num_features == '1c':
        num_features = 1
        feature = 'correct'
    elif num_features == '1m':
        num_features = 1
        feature = 'master'
    elif num_features == '1l':
        num_features = 1
        feature = 'length'
    else:
        num_features = int(num_features)
    
    load_data = LoadData()
    slide_sequences, slide_labels = load_data.get_RNN_data()
    num_incorrect, easy_slides, hard_slides, correct_images = load_data.get_other_var()
    
    baseline_model = BaselineModel()
    baseline_model.train(slide_sequences, slide_labels, correct_images, num_features, feature)
    
    if args.conf:
        print('DISPLAYING CONFUSION MATRICES:\n')
        conf = baseline_model.get_confusion_matrices()
        for slide_id in range(NUM_SLIDES):  
            print('Slide '+str(slide_id+1)+':\n', conf[slide_id])

    if args.acc:
        print('DISPLAYING ACCURACIES:\n')
        acc = baseline_model.get_accuracies()
        for slide_id in range(NUM_SLIDES):
            print('Slide '+str(slide_id+1)+':', acc[slide_id], '\n')
        print('Average accuracy of slide 2 to 21:', acc[1:].mean())
    
    if args.auc:
        print('DISPLAYING AUC SCORES:\n')
        auc = baseline_model.get_auc_scores()
        for slide_id in range(NUM_SLIDES):
            print('Slide '+str(slide_id+1)+':', auc[slide_id], '\n')
        print('Average auc score of slide 2 to 21:', auc[1:].mean())