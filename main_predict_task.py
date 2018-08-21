import os
import pickle
import argparse

from load_data.load_data import LoadData
from network.network import Network
from utils.ablation_utils import modify_sequences_in_slide
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed) 

NUM_SLIDES = 21
NUM_IMAGES = 6
k = 5

if __name__ == '__main__':
    description = 'Show graphs of probability of choosing each image vs number of fixations for all 21 slides\n'+\
    'Requirements:\n'+\
    '    - All data must be available for LoadData() to work (for more information, see the description of main_network_choice.py)\n'+\
    '    - The SELECTION networks have already been trained and saved (for more information, see main_network_choice.py)'
    
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--from_scratch', help='Calculate and save all data for probability vs fixations for each slide from scratch' + \
        'Skip if already calculated and saved before', action="store_true")
    parser.add_argument('--plot_prob', help='Plot the graphs of probability of choosing each image vs. fixations', action="store_true")
    parser.add_argument('--plot_acc', help='Plot the graphs of prediction accuracy of the network vs. fixations', action="store_true")

    args = parser.parse_args()
    
    # Load data
    load_data = LoadData()
    slide_sequences, slide_selections = load_data.get_RNN_data_selection()
    max_sequence_lengths = load_data.get_max_sequence_lengths()
    NUM_SUBJECTS = slide_selections.shape[1]
    _, _, _, corr_imgs = load_data.get_other_var()
    
    if args.from_scratch:
        #Load models
        network = Network()
        network.load_models("SELECTION")
        models = network.get_models()   #NUM_SLIDES x k model (each slide has k folds)

        stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
        # a database of all probability vs fixation for all slides and all images in each slide
        # e.g. all_prob_fix_arr[20][5] will show the whole progression of probability vs fixation for slide 21, image 6
        all_prob_fix_arr = []
        
        # a database of all prediction accuracy vs fixation for all slides
        # e.g. all_acc_arr[20] returns an array of max_sequence_lengths[20] elements
        all_acc_fix_arr = []

        for slide_id in range(NUM_SLIDES):
            sequence = slide_sequences[slide_id]   #NUM_SUBJECTS sequences
            mod = models[slide_id]
            max_len = max_sequence_lengths[slide_id]
            y = slide_selections[slide_id] - 1
            
            # Calculate the probabilities of choosing each of the 6 images with m first fixations (0 <= m <= max_len)
            # Probabilities in this case means the average probability of all subjects
            prob_fix_arr = np.zeros((max_len+1, NUM_IMAGES))    #Will take transpose at the end so that prob of choosing image i with m last fixations is at element (i,m)
            
            acc_fix_arr = np.zeros(max_len+1)
            
            for num_first_fix in range(max_len+1):
                ablated_sequence = modify_sequences_in_slide(sequence, num_last_arg=None, mode='keep', section='pre', num_first_arg=num_first_fix)
                X = pad_sequences(ablated_sequence, maxlen=max_len)
                prob = np.zeros(6)
                acc = 0.
                fold = 0
                for train,test in stratified_kfold.split(X,y):
                    fold_prob = mod[fold].predict(X[test])
                    prob += np.sum(fold_prob, axis=0)
                    fold_pred = mod[fold].predict_classes(X[test], verbose=0).flatten()
                    num_pred_corr = len(test) - np.count_nonzero(y[test]-fold_pred)
                    acc += num_pred_corr
                    fold += 1
                prob = prob / float(NUM_SUBJECTS)
                prob_fix_arr[num_first_fix] = prob
                acc = acc / float(NUM_SUBJECTS)
                acc_fix_arr[num_first_fix] = acc
            prob_fix_arr = prob_fix_arr.T
            all_prob_fix_arr.append(prob_fix_arr)
            all_acc_fix_arr.append(acc_fix_arr)
        
        # make directories:
        if not os.path.exists('network/probability_vs_fixation'):
	    os.mkdir('network/probability_vs_fixation')
	if not os.path.exists('network/accuracy_vs_fixation'):
	    os.mkdir('network/accuracy_vs_fixation')
        # save data
        with open('network/probability_vs_fixation/prob_vs_fix.pickle', 'wb') as pickle_file:
            pickle.dump(all_prob_fix_arr, pickle_file)
        with open('network/accuracy_vs_fixation/acc_vs_fix.pickle', 'wb') as pickle_file:
            pickle.dump(all_acc_fix_arr, pickle_file)
    
    else:
        all_prob_fix_arr = pickle.load(open('network/probability_vs_fixation/prob_vs_fix.pickle', 'rb'))
        all_acc_fix_arr = pickle.load(open('network/accuracy_vs_fixation/acc_vs_fix.pickle', 'rb'))
    
    if args.plot_prob:
        for slide_id in range(NUM_SLIDES):
            max_len = max_sequence_lengths[slide_id]
            corr_img = corr_imgs[slide_id]
            prob_fix_arr = all_prob_fix_arr[slide_id]
            
            # Plot the probabilities against number of last fixations
            plt.figure()
            plt.xlabel('Number of fixations in sequence', fontsize=15)
            x_axis = np.arange(max_len+1)
            plt.ylabel('Probability of choosing each slide', fontsize=15)
            legend_names = []

            for img in range(1, NUM_IMAGES+1):
                y_axis = prob_fix_arr[img-1]
                plt.plot(x_axis, y_axis)
                legend = 'Image ' + str(img)
                if img+1 == corr_img:
                    legend += ' (correct image)'
                legend_names.append(legend)
            plt.legend(legend_names, borderpad=3, labelspacing=2, prop={'size': 15})
            plt.title('Slide '+str(slide_id+1), fontsize=30)
            
    if args.plot_acc:
        for slide_id in range(NUM_SLIDES):
            max_len = max_sequence_lengths[slide_id]
            acc_fix_arr = all_acc_fix_arr[slide_id]
            
            # Plot the probabilities against number of last fixations
            plt.figure()
            plt.xlabel('Number of fixations in sequence', fontsize=15)
            x_axis = np.arange(max_len+1)
            plt.ylabel('Prediction accuracy', fontsize=15)
            y_axis = acc_fix_arr
            plt.plot(x_axis, y_axis)
            plt.title('Slide '+str(slide_id+1), fontsize=30)



