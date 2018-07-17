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
    parser.add_argument('--plot', help='Plot the graphs', action="store_true")

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
        
        # a database of all probability vs fixation data for all slides and all images in each slide
        # e.g. all_prob_fix_arr[20][5] will show the whole progression of probability vs fixation for slide 21, image 6
        all_prob_fix_arr = []

        for slide_id in range(NUM_SLIDES):
            sequence = slide_sequences[slide_id]   #NUM_SUBJECTS sequences
            mod = models[slide_id]
            max_len = max_sequence_lengths[slide_id]
            y = slide_selections[slide_id]
            
            # Calculate the probabilities of choosing each of the 6 images with m last fixations (1 <= m <= max_len)
            # Probabilities in this case means the average probability of all subjects
            prob_fix_arr = np.zeros((max_len, NUM_IMAGES))    #Will take transpose at the end so that prob of choosing image i with m last fixations is at element (i,m)
            for num_last_fix in range(1, max_len+1):
                post_pad_sequence = pad_sequences(sequence,padding='post')
                ablated_sequence = modify_sequences_in_slide(post_pad_sequence, num_last_fix, 'discard', 'post')
                X = pad_sequences(ablated_sequence, maxlen=max_len)
                prob = np.zeros(6)
                fold = 0
                for train,test in stratified_kfold.split(X,y):
                    fold_prob = mod[fold].predict(X[test])
                    prob += np.sum(fold_prob, axis=0)
                    fold += 1
                prob = prob / NUM_SUBJECTS
                prob_fix_arr[num_last_fix-1] = prob
            prob_fix_arr = prob_fix_arr.T
            all_prob_fix_arr.append(prob_fix_arr)
        
        # make directories:
	    if not os.path.exists('network/probability_vs_fixation'):
		    os.mkdir('network/probability_vs_fixation')
        # save data
        with open('network/probability_vs_fixation/prob_vs_fix.pickle', 'wb') as pickle_file:
            pickle.dump(all_prob_fix_arr, pickle_file)
    
    else:
        all_prob_fix_arr = pickle.load(open('network/probability_vs_fixation/prob_vs_fix.pickle', 'rb'))
    
    
    if args.plot:
        for slide_id in range(NUM_SLIDES):
            max_len = max_sequence_lengths[slide_id]
            corr_img = corr_imgs[slide_id]
            prob_fix_arr = all_prob_fix_arr[slide_id]
            
            # Plot the probabilities against number of last fixations
            plt.figure()
            plt.xlabel('Number of last fixations discarded', fontsize=15)
            x_axis = np.arange(1, max_len+1)
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



