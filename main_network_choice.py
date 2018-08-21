from load_data.load_data import LoadData
from network.network import Network
from create_dataframe.create_dataframe import CreateDataframe
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from utils.ablation_utils import modify_all_sequences

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed) 

NUM_SLIDES = 21

if __name__ == '__main__':
    description = 'Show result of network and output dataframes to csv files in the create_dataframe directory\n'+\
    'Requirements: The raw_data directory must contain:\n'+\
    '    -  A "mfft_sequence.txt" that must not have header and contains the following columns: '+\
    'Subject, ImId, SelectionOrderAtt, correctVariant(s), FixationStart, '+\
    'FixationFrequencyPerChoiceInterval, TimeToFirstSelectionAtt\n'+\
    '    -  A "Concussion_data.csv" file that contains various measures\n'+\
    '    -  A "CC& CINcomponents& SLFcomponents.xlsx" file that contains additional DTI measures\n'+\
    'See the raw_data director for examples of these files'
    
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--train', help='Train and save the network from scratch. ' + \
        'Skip if already have network trained and saved prior', action="store_true")
    parser.add_argument('--conf', help='Show confusion matrices', action="store_true")
    parser.add_argument('--acc', help='Show accuracies', action="store_true")
    parser.add_argument('--auc', help='Show auc_scores', action="store_true")
    parser.add_argument('--plot_val_loss', help='Plot validation losses', action="store_true")
    parser.add_argument('--output', help='Output dataframes to csv files', action="store_true")
    
    parser.add_argument('--discard_last_dup', help='Discard all last fixations', action="store_true")
    
    args = parser.parse_args()
    
    load_data = LoadData()
    slide_sequences, slide_selections = load_data.get_RNN_data_selection()

    network = Network()
    
    if args.discard_last_dup:
        discard_last_dup_sequences = modify_all_sequences(slide_sequences, num_last_arg=None, mode='discard last dup', section=None)
        print('slide_sequences[0][0]: ', slide_sequences[0][0])
        print('discard_last_dup_sequences[0][0]: ', discard_last_dup_sequences[0][0])
        print('slide_sequences[9][9]: ', slide_sequences[9][9])
        print('discard_last_dup_sequences[9][9]: ', discard_last_dup_sequences[9][9])
        slide_sequences = discard_last_dup_sequences
    
    if args.train:
        network.train_and_save(slide_sequences, slide_selections, "SELECTION", should_save_auc=False)
    else:
        network.load_results("SELECTION")
    
    if args.conf:
        print('DISPLAYING CONFUSION MATRICES:\n')
        conf = network.get_confusion_matrices()
        for slide_id in range(NUM_SLIDES):  
            print('Slide '+str(slide_id+1)+':\n', conf[slide_id])
    
    if args.acc:
        print('DISPLAYING ACCURACIES:\n')
        acc = network.get_accuracies()
        for slide_id in range(NUM_SLIDES):
            print('Slide '+str(slide_id+1)+':', acc[slide_id], '\n')
        print('Average accuracy of slide 2 to 21:', acc[1:].mean())
    
    if args.auc:
        print('DISPLAYING AUC SCORES:\n')
        auc = network.get_auc_scores()
        for slide_id in range(NUM_SLIDES):
            print('Slide '+str(slide_id+1)+':', auc[slide_id], '\n')
        print('Average auc score of slide 2 to 21:', auc[1:].mean())
    
    if args.plot_val_loss:
        network.load_histories()
        network.plot_val_loss()
    
    if args.output:
        subject_array = load_data.get_subject_array()
        num_incorrect,easy_slides,hard_slides,correct_images = load_data.get_other_var()
        y_probability = network.get_y_probability()
        create_dataframe = CreateDataframe(subject_array, slide_labels, easy_slides, hard_slides, y_probability)
        create_dataframe.output_to_csv()
