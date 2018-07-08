import numpy as np
NUM_SLIDES = 21
NUM_IMAGES = 7  #images per slide, including maxster image

def get_fixation_sequence_from_time(slide_times):
    '''
    Param:
        slide_times: array of array of time. E.g.
                        895.7,895.867,898.533,
                                        897.267,
                                        896.967,
                                            -1
                                        897.467,
                            895.133,895.5,897.8,
        898.067,898.867,899.5,899.867,900.033,
    Return:
    sequence: Sequence of fixation. Type: numpy array. E.g.
        [6, 6, 1, 1, 3, 2, 5, 6, 7, 1, 7, 7, 7, 7]
    '''
    sequence = []
    all_list_empty = 0
    while all_list_empty == 0:
        all_list_empty = 1
        min_time = 999999999
        next_img = None
        for i in range(NUM_IMAGES):
            if len(slide_times[i]) != 0:
                if slide_times[i][0] < min_time:
                    min_time = slide_times[i][0]
                    next_img = i+1
                all_list_empty = 0
        if next_img != None:
            del(slide_times[next_img-1][0])
            sequence += [next_img]
    return np.array(sequence)

class Slide():
    '''
    Attributes:
        Sequence
        IsCorrect
    '''
    def __init__(self, sequence, is_correct):
        self.Sequence = sequence
        self.IsCorrect = is_correct

class Subject():
    '''
    Attributes:
        Name
        SlideArray: array of Slide
    '''
    def __init__(self, name, slide_array):
        self.Name = name
        self.SlideArray = slide_array
    
    #get all 21 sequences of this subject
    def get_sequences(self):
        sequences = []
        for slide in self.SlideArray:
            sequences.append(slide.Sequence)
        return sequences
    
    def get_is_corrects(self):
        is_corrects = []
        for slide in self.SlideArray:
            is_corrects.append(slide.IsCorrect)
        return is_corrects

def normalize_subject_name(subject_name):
    subject_name = subject_name.lower()
    if 'p' in subject_name or '-c' in subject_name:
        if 'control' in subject_name:
            subject_name = subject_name[:-8]
        if 'obi' in subject_name:
            subject_name = subject_name[:-4]
    return subject_name    