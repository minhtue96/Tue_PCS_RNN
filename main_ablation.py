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

if __name__ == '__main__':
    description = 'Ablate the sequences, input the ablated sequences into the network '+\
        'and show the result'

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    help_method =   "1. Shuffle whole sequence\n"+\
                    "2. Keep only 1st half\n"+\
                    "3. Keep only 2nd half\n"+\
                    "4. Shuffle 1st half and keep 2nd half\n"+\
                    "5. Shuffle 2nd half and keep 1st half\n"+\
                    "6. Discard 1st half and shuffle 2nd half\n"+\
                    "7. Discard last 5 fixations"
    parser.add_argument('method', choices=range(1,8), help=help_method, type=int)
    
    parser.add_argument('--conf', help='Show confusion matrices', action="store_true")
    parser.add_argument('--acc', help='Show accuracies', action="store_true")
    parser.add_argument('--auc', help='Show auc_scores', action="store_true")
    
    load_data = LoadData()
    slide_sequences, slide_labels = load_data.get_RNN_data()
    max_sequence_lengths = load_data.get_max_sequence_lengths()

    args = parser.parse_args()
    
    network = Network()
    network.load_models()
    model = network.get_models()
    
    method = int(args.method)
    if method == 1:
        ablated_sequences = modify_all_sequences(slide_sequences, 0, 'shuffle', 'post')
    elif method == 2:
        ablated_sequences = modify_all_sequences(slide_sequences, 'half', 'discard', 'post')
    elif method == 3:
        ablated_sequences = modify_all_sequences(slide_sequences, 'half', 'discard', 'pre')
    elif method == 4:
        ablated_sequences = modify_all_sequences(slide_sequences, 'half', 'shuffle', 'pre')
    elif method == 5:
        ablated_sequences = modify_all_sequences(slide_sequences, 'half', 'shuffle', 'post')
    elif method == 6:
        ablated_sequences = modify_all_sequences(slide_sequences, 'half', 'shuffle', 'post')    #shuffle second half
        ablated_sequences = modify_all_sequences(ablated_sequences, 'half', 'discard', 'pre')   #discard first half
    elif method == 7:
        ablated_sequences = modify_all_sequences(slide_sequences, 5, 'discard', 'post')
    
    ablation = Ablation(slide_labels, model, max_sequence_lengths, ablated_sequences)
    
    if args.conf:
        print('DISPLAYING CONFUSION MATRICES:\n')
        conf = ablation.get_confusion_matrices()
        for slide_id in range(NUM_SLIDES):  
            print('Slide '+str(slide_id+1)+':\n', conf[slide_id])

    if args.acc:
        print('DISPLAYING ACCURACIES:\n')
        acc = ablation.get_accuracies()
        for slide_id in range(NUM_SLIDES):
            print('Slide '+str(slide_id+1)+':', acc[slide_id], '\n')
        print('Average accuracy of slide 2 to 21:', acc[1:].mean())
    
    if args.auc:
        print('DISPLAYING AUC SCORES:\n')
        auc = ablation.get_auc_scores()
        for slide_id in range(NUM_SLIDES):
            print('Slide '+str(slide_id+1)+':', auc[slide_id], '\n')
        print('Average auc score of slide 2 to 21:', auc[1:].mean())