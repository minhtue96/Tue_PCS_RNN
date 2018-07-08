import numpy as np
import pandas as pd
import copy
from utils.load_data_utils import Slide, Subject
from utils.load_data_utils import get_fixation_sequence_from_time, normalize_subject_name
np.random.seed(7)

NUM_SLIDES = 21
NUM_IMAGES = 7  #images per slide, including maxster image

class LoadData:
	def __init__(self):
		
		text = pd.read_csv('raw_data/mfft_sequence2.txt',delimiter='\t')

		for line_id in range(len(text)):
			fix_freq_arr = text.loc[line_id,'FixationFrequencyPerChoiceInterval'].split(',')[:-2]
			fix_freq_to_1st_choice = int(fix_freq_arr[0])
			text.set_value(line_id,'FixationFrequencyPerChoiceInterval',fix_freq_to_1st_choice)
			
			fix_start_arr = text.loc[line_id,'FixationStart'].split(',')[:-1]
			fix_start_arr = [float(fix_start) for fix_start in fix_start_arr]
			if fix_freq_to_1st_choice == -1:
				text.set_value(line_id,'FixationStart',[])
			else:
				text.set_value(line_id,'FixationStart',fix_start_arr[:fix_freq_to_1st_choice])
					
		#e.g. text[0] = array(['01-005-P2 OBI', 1, 0, 'MasterImage', [895.7, 895.867, 898.533]], dtype=object)
		NUM_SUBJECTS = int(len(text) / (NUM_SLIDES*NUM_IMAGES))
		NUM_SEQUENCES = NUM_SLIDES*NUM_SUBJECTS #total number of all sequences generated
		
		subject_array = []
		X_not_pad = []
		y_all = []
		selection_all = []
		for i in range(NUM_SUBJECTS):   #each i correspond to 147 lines
			name = text.loc[i*NUM_IMAGES*NUM_SLIDES,'Subject']
			name = normalize_subject_name(name)
			slide_array = []
			for j in range(NUM_SLIDES): #each j corresponds to 7 lines
				start_slide_id = i*NUM_SLIDES*NUM_IMAGES + j*NUM_IMAGES
				end_slide_id = i*NUM_SLIDES*NUM_IMAGES + (j+1)*NUM_IMAGES
				slide_times = copy.deepcopy(text.loc[start_slide_id:end_slide_id, 'FixationStart'].values)
				sequence = get_fixation_sequence_from_time(slide_times)
				
				is_correct = None
				selected = None
				for k in range(NUM_IMAGES):
					id = start_slide_id+k
					if text.loc[id, 'SelectionOrderAtt'] == 1:
						is_correct = 1 if (text.loc[id,'correctVariant(s)'] == 'correct') else 0
						selected = k
						break
				
				slide = Slide(sequence, is_correct)
				slide_array.append(slide)
				
				X_not_pad.append(sequence)
				y_all.append(is_correct)
				selection_all.append(selected)
					
			subject = Subject(name,slide_array)
			subject_array.append(subject)
		X_not_pad = np.array(X_not_pad)
		y_all = np.array(y_all)
		selection_all = np.array(selection_all)		
		
		correct_images = []
		for line_id in range(NUM_IMAGES*NUM_SLIDES):
			if text.loc[line_id,'correctVariant(s)'] == 'correct':
				correct_images.append(text.loc[line_id,'ImId'])
		correct_images = np.array(correct_images)
		
		slide_sequences = []	#list of 21 elmt, each is list of sequences for each slide
								#e.g. slide_sequences[0] is the list of sequences of 1st slide for all subjects
		slide_labels = []
		slide_selections = []
		for i in range(NUM_SLIDES):
			slide_sequences.append([])
			slide_labels.append([])
			slide_selections.append([])
		
		for i in range(NUM_SEQUENCES):
			slide_id = i%NUM_SLIDES
			slide_sequences[slide_id].append(X_not_pad[i])
			slide_labels[slide_id].append(y_all[i])
			slide_selections[slide_id].append(selection_all[i])

		slide_sequences = np.array(slide_sequences)
		slide_labels = np.array(slide_labels)
		slide_selections = np.array(slide_selections)

		num_incorrect = np.zeros(NUM_SLIDES, dtype=int)	#total number of incorrect answers for each slide.
		for i in range(NUM_SLIDES):
			num_incorrect[i] = NUM_SUBJECTS-slide_labels[i].sum()
		
		hard_slides = np.sort(np.argsort(num_incorrect)[11:])
		easy_slides = np.sort(np.argsort(num_incorrect)[1:11])
		
		self.slide_sequences = slide_sequences
		self.slide_labels = slide_labels
		self.slide_selections = slide_selections
		self.num_incorrect = num_incorrect
		self.easy_slides = easy_slides
		self.hard_slides = hard_slides
		self.correct_images = correct_images
		self.subject_array = np.array(subject_array)
	
	
	def get_RNN_data(self):
		'''
		return:
			slide_sequences: list of 21 elements, each is list of sequences for each slide.
			slide_labels: list of 21 elements, each is list of correctnes of selection for each slide
		'''
		return self.slide_sequences, self.slide_labels

	def get_RNN_data_selection(self):
		'''
		return:
			slide_sequences: list of 21 elements, each is list of sequences for each slide.
			slide_selections: list of 21 elements, each is list of selection for each slide
		'''
		return self.slide_sequences, self.slide_selections
		
	def get_other_var(self):
		'''
		return:
			num_incorrect: number of incorrect selections for each slide. Used as metric for difficulty of slides
			easy_slides: index of 10 easiest slides (not considering slide 1)
			hard_slides: index of 10 hardest slides
			correct_images: list of 21 elements, each is the correct image of each slide
		'''
		return self.num_incorrect, self.easy_slides, self.hard_slides, self.correct_images
		
	
	def get_subject_array(self):
		'''
		return:
			subject_array: list of Subjects objects
		'''
		return self.subject_array
		
	def get_max_sequence_lengths(self):
		'''
		return:
			max_sequence_lengths: list of 21 elements, each is the maximum lengths of the sequences in each slide
		'''
		max_sequence_lengths = []
		for slide_id in range(NUM_SLIDES):
			max_sequence_lengths.append(len(max(self.slide_sequences[slide_id], key=len)))
		return np.array(max_sequence_lengths)
		
