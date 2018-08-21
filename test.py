'''
from load_data.load_data import LoadData
from utils.ablation_utils import modify_sequences_in_slide
import numpy as np
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed) 

NUM_SLIDES = 21
NUM_IMAGES = 6
k = 5


load_data = LoadData()
slide_sequences, slide_selections = load_data.get_RNN_data_selection()
max_sequence_lengths = load_data.get_max_sequence_lengths()
NUM_SUBJECTS = slide_selections.shape[1]

sequence = slide_sequences[0]
max_len = max_sequence_lengths[0]

num_first_arg = max_len
ablated_sequence = modify_sequences_in_slide(sequence, num_last_arg=None, mode='keep', section='pre', num_first_arg=num_first_arg)
X = pad_sequences(ablated_sequence, maxlen=max_len)
'''

'''
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
        network.load_models("CORRECT")
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
            prob_fix_arr = np.zeros((max_len, 2))    #Will take transpose at the end so that prob of choosing image i with m last fixations is at element (i,m)
            for num_last_fix in range(1, max_len+1):
                post_pad_sequence = pad_sequences(sequence,padding='post')
                ablated_sequence = modify_sequences_in_slide(post_pad_sequence, num_last_fix, 'discard', 'post')
                X = pad_sequences(ablated_sequence, maxlen=max_len)
                prob = np.zeros(2)
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
        with open('network/probability_vs_fixation/prob_vs_fix_corr.pickle', 'wb') as pickle_file:
            pickle.dump(all_prob_fix_arr, pickle_file)
    
    else:
        all_prob_fix_arr = pickle.load(open('network/probability_vs_fixation/prob_vs_fix_corr.pickle', 'rb'))
    
    
    if args.plot:
        for slide_id in range(NUM_SLIDES):
            max_len = max_sequence_lengths[slide_id]
            corr_img = corr_imgs[slide_id]
            prob_fix_arr = all_prob_fix_arr[slide_id]
            
            # Plot the probabilities against number of last fixations
            plt.figure()
            plt.xlabel('Number of last fixations discarded', fontsize=15)
            x_axis = np.arange(1, max_len+1)
            plt.ylabel('Probability of correct/incorrect', fontsize=15)
            legend_names = []

            for is_corr in range(2):
                y_axis = prob_fix_arr[is_corr]
                plt.plot(x_axis, y_axis)
                legend_names = ['incorrect', 'correct']
            plt.legend(legend_names, borderpad=3, labelspacing=2, prop={'size': 15})
            plt.title('Slide '+str(slide_id+1), fontsize=30)
'''

'''
from load_data.load_data import LoadData

import os
import numpy as np
import tensorflow as tf

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed) 

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

slide_id = 0
fold = 0
#load_models
print('Load model for slide', slide_id+1, 'fold', fold+1)
filename = 'network/models{}/model{}_{}.h5'.format(
	'SELECTION', slide_id+1, fold+1)
model = load_model(filename)

load_data = LoadData()
slide_sequences, slide_selections = load_data.get_RNN_data_selection()
sequence = slide_sequences[slide_id]
selection = slide_selections[slide_id]


y_pred = model.predict_classes(pad_sequences(sequence), verbose=0)
y_test = selection-1

conf_matrix = confusion_matrix(y_test, y_pred)

conf_dict = {}
for i in range(0, conf_matrix.shape[0]):
	for j in range(0, conf_matrix.shape[1]):
		i_val = np.unique(np.append(y_test, y_pred))[i]
		j_val = np.unique(np.append(y_test, y_pred))[j]
		conf_dict[i_val, j_val] = conf_matrix[i, j]

new_conf_mat = np.zeros(shape=(6, 6))
for i in range(0, 6):
	for j in range(0, 6):

		if (i, j) not in conf_dict:
			new_conf_mat[i, j] = 0
		else:

			new_conf_mat[i, j] = conf_dict[(i, j)]
'''
'''
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
'''
'''
stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

slide_id = 0
k=5
X=pad_sequences(ablated_sequences[slide_id], maxlen=max_sequence_lengths[slide_id])
y = slide_labels[slide_id]
y=y-1
fold=0
for train,test in stratified_kfold.split(X,y):
    y_pred = model[slide_id,fold].predict_classes(X[test], verbose=0).flatten()
    accuracy_score(y[test],y_pred)
    fold += 1
'''

'''
#plot graphs of probability vs fixations (can limit within range of sequence length. can work with absolute fixations or fraction of sequence). Use to see if can predict

from load_data.load_data import LoadData
from network.network import Network
from utils.ablation_utils import modify_sequences_in_slide

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences

seed=7

model_type_name = 'SELECTION'

load_data = LoadData()
slide_sequences, slide_selections = load_data.get_RNN_data_selection()
NUM_SUBJECTS = slide_selections.shape[1]
NUM_SLIDES=21
NUM_IMAGES = 6
k = 5
network = Network()
max_sequence_lengths = load_data.get_max_sequence_lengths()
_, _, _, corr_imgs = load_data.get_other_var()
stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

slide_id = 9
model = network.load_and_get_model_slide(model_type_name, slide_id)
slide_sequence = slide_sequences[slide_id]
slide_selection = slide_selections[slide_id] - 1
slide_sequence_len = np.array([len(slide_sequence[subject]) for subject in range(NUM_SUBJECTS)])
max_len = max_sequence_lengths[slide_id]

num_div = 20        #inspect each sequence at 0, 1/num_div, 2/num_div, ... full sequence

lower_len = 100
upper_len = max_len
subjects_chosen = np.where((slide_sequence_len>=lower_len) & (slide_sequence_len<=upper_len))[0]      #subjects that have sequence length between lower_len and upper_len
num_subjects_chosen = len(subjects_chosen)

test_subjects_folds = []
for train,test in stratified_kfold.split(slide_sequence,slide_selection):
    test_subjects_folds.append(test)

subjects_chosen_in_folds = [[] for fold in range(k)]
for subject_chosen in subjects_chosen:
    for fold in range(k):
        if subject_chosen in test_subjects_folds[fold]:
            subjects_chosen_in_folds[fold].append(subject_chosen)
            break

prob_fix_arr = np.zeros((lower_len+1,NUM_IMAGES))
acc_fix_arr = np.zeros((lower_len+1))

#prob_fix_arr = np.zeros((num_div+1,NUM_IMAGES))
#acc_fix_arr = np.zeros((num_div+1))

for num_first_fix in range(lower_len+1):
    ablated_sequence = modify_sequences_in_slide(slide_sequence, num_last_arg=None, mode='keep', section='pre', num_first_arg=num_first_fix)
#for div in range(num_div+1):
#    ablated_sequence = modify_sequences_in_slide(slide_sequence, num_last_arg=None, mode='keep', section='pre', num_first_arg=float(div)/num_div)
    X = pad_sequences(ablated_sequence, maxlen=max_len)
    prob = np.zeros(NUM_IMAGES)
    acc = 0.
    for fold in range(k):
        subjects_chosen_in_fold = subjects_chosen_in_folds[fold]
        if subjects_chosen_in_fold == []:
            continue
        fold_prob = model[fold].predict(X[subjects_chosen_in_fold])
        prob += np.sum(fold_prob, axis=0)
        fold_pred = model[fold].predict_classes(X[subjects_chosen_in_fold], verbose=0).flatten()
        num_pred_corr = len(subjects_chosen_in_fold) - np.count_nonzero(slide_selection[subjects_chosen_in_fold]-fold_pred)
        acc += num_pred_corr
    prob = prob / float(num_subjects_chosen)
    prob_fix_arr[num_first_fix] = prob
    #prob_fix_arr[div] = prob
    acc = acc / float(num_subjects_chosen)
    acc_fix_arr[num_first_fix] = acc
    #acc_fix_arr[div] = acc
prob_fix_arr = prob_fix_arr.T


corr_img = corr_imgs[slide_id]
# Plot the probabilities against number of last fixations
plt.figure()
plt.xlabel('Number of fixations in sequence', fontsize=15)
x_axis = np.arange(lower_len+1)
#plt.xlabel('Fraction of sequence', fontsize=15)
#x_axis = np.arange(num_div+1)/float(num_div)
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

np.sort(slide_sequence_len)
'''

'''
from load_data.load_data import LoadData
from network.network import Network
from utils.ablation_utils import modify_all_sequences
from ablation.ablation import Ablation
import numpy as np
import tensorflow as tf
import argparse

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed) 

NUM_SLIDES = 21
k = 5

    
model_type_name = "SELECTION"

load_data = LoadData()
if model_type_name == "SELECTION":
    slide_sequences, slide_labels = load_data.get_RNN_data_selection()
else:
    slide_sequences, slide_labels = load_data.get_RNN_data()
max_sequence_lengths = load_data.get_max_sequence_lengths()

network = Network()
slide_id = 9
mod = network.load_and_get_model_slide(model_type_name, slide_id)
model = [[0 for fold in range(k)] for _ in range(NUM_SLIDES)]
model[slide_id] = mod
model = np.array(model)

ablated_sequences = modify_all_sequences(slide_sequences, 1, 'discard', 'post')

ablation = Ablation(slide_labels, model, max_sequence_lengths, ablated_sequences, model_type_name, ablated_slide_id=slide_id)




print('DISPLAYING ACCURACIES:\n')
acc = ablation.get_accuracies()[slide_id]
print('Slide '+str(slide_id+1)+':', acc, '\n')
'''

#plot graphs of accuracy vs fixations discarded at end 

from load_data.load_data import LoadData
from network.network import Network
from utils.ablation_utils import modify_sequences_in_slide
from baseline_model.baseline_model import BaselineModel

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences

seed=7

model_type_name = 'SELECTION'

load_data = LoadData()
slide_sequences, slide_selections = load_data.get_RNN_data() if (model_type_name == 'CORRECT') else load_data.get_RNN_data_selection()
NUM_SUBJECTS = slide_selections.shape[1]
NUM_SLIDES=21
NUM_IMAGES = 6
k = 5
network = Network()
max_sequence_lengths = load_data.get_max_sequence_lengths()
_, _, _, corr_imgs = load_data.get_other_var()
baseline_model = BaselineModel()
baseline_model.train(slide_sequences, slide_selections, corr_imgs, num_features=6, feature=None)
stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

slide_id = 16
model = network.load_and_get_model_slide(model_type_name, slide_id)
slide_sequence = slide_sequences[slide_id]
slide_selection = slide_selections[slide_id] if (model_type_name == 'CORRECT') else slide_selections[slide_id] - 1
slide_sequence_len = np.array([len(slide_sequence[subject]) for subject in range(NUM_SUBJECTS)])
max_len = max_sequence_lengths[slide_id]

lower_len = 50
upper_len = max_len
subjects_chosen = np.where((slide_sequence_len>=0) & (slide_sequence_len<=upper_len))[0]      #subjects that have sequence length between lower_len and upper_len
num_subjects_chosen = len(subjects_chosen)

test_subjects_folds = []
for train,test in stratified_kfold.split(slide_sequence,slide_selection):
    test_subjects_folds.append(test)

subjects_chosen_in_folds = [[] for fold in range(k)]
for subject_chosen in subjects_chosen:
    for fold in range(k):
        if subject_chosen in test_subjects_folds[fold]:
            subjects_chosen_in_folds[fold].append(subject_chosen)
            break

ablated_sequences = [[] for slide in range(NUM_SLIDES)]
acc_fix_arr = np.zeros((lower_len+1))
baseline_acc_fix_arr = np.zeros((lower_len+1))


for num_last_discarded in range(lower_len+1):
    ablated_sequence = modify_sequences_in_slide(slide_sequence, num_last_arg=num_last_discarded, mode='discard', section='post')
    ablated_sequences[slide_id] = ablated_sequence
    if num_last_discarded%10==0:
        print(ablated_sequence[0])
    X = pad_sequences(ablated_sequence, maxlen=max_len)
    acc = 0.
    for fold in range(k):
        subjects_chosen_in_fold = subjects_chosen_in_folds[fold]
        if subjects_chosen_in_fold == []:
            continue
        fold_pred = model[fold].predict_classes(X[subjects_chosen_in_fold], verbose=0).flatten()
        num_pred_corr = len(subjects_chosen_in_fold) - np.count_nonzero(slide_selection[subjects_chosen_in_fold]-fold_pred)
        acc += num_pred_corr
    acc = acc / float(num_subjects_chosen)
    acc_fix_arr[num_last_discarded] = acc
    if num_last_discarded == 5:
        print('acc when discard 5 last fix:',acc)
    
    
    
    
    baseline_acc = baseline_model.calculate_slide_ablated_acc(ablated_sequences, slide_selections, slide_id, subjects_chosen_in_folds)
    baseline_acc_fix_arr[num_last_discarded] = baseline_acc


corr_img = corr_imgs[slide_id]
# Plot the probabilities against number of last fixations
plt.figure()
plt.xlabel('Number of fixations discarded at the end', fontsize=15)
x_axis = -np.arange(lower_len+1)
plt.ylabel('Accuracy', fontsize=15)
plt.plot(x_axis, acc_fix_arr)
plt.plot(x_axis, baseline_acc_fix_arr)
legend_names = ['RNN', 'Baseline']

plt.legend(legend_names, borderpad=3, labelspacing=2, prop={'size': 15})
plt.title('Slide '+str(slide_id+1), fontsize=30)

np.sort(slide_sequence_len)








'''
for num_last_discarded in range(1,6):
    ablated_sequence = modify_sequences_in_slide(slide_sequence, num_last_arg=num_last_discarded, mode='discard', section='pre')
    X = pad_sequences(ablated_sequence, maxlen=max_len)
    acc = 0.
    for fold in range(k):
        subjects_chosen_in_fold = subjects_chosen_in_folds[fold]
        if subjects_chosen_in_fold == []:
            continue
        fold_pred = model[fold].predict_classes(X[subjects_chosen_in_fold], verbose=0).flatten()
        num_pred_corr = len(subjects_chosen_in_fold) - np.count_nonzero(slide_selection[subjects_chosen_in_fold]-fold_pred)
        acc += num_pred_corr
    acc = acc / float(num_subjects_chosen)
    print('num last keep:', num_last_discarded, 'acc:',acc)
'''

