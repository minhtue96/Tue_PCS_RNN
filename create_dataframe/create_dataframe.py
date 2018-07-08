import numpy as np
import pandas as pd

from utils.load_data_utils import normalize_subject_name
NUM_SLIDES = 21
NUM_IMAGES = 7

'''
Slide 1 is not counted when taking measures on easy, hard and all slides
'''

class CreateDataframe:
    '''
    Attributes:
        df: dataframe of all subjects (patients, control, cte; including both subjects doing MFFT and subjects not)
        df_many: dataframe of only patients with more than 1 concussion
    '''
    
    def __init__(self, subject_array, slide_labels, easy_slides, hard_slides, y_probability):
        '''
        Arguments:
            subject_array: array of Subject objects, shape (NUM_SUBJECTS,) i.e. only subjects going through MFFT. Can get from LoadData
            slide_labels: Correctness of subjects' selections, shape (NUM_SLIDES, NUM_SUBJECTS). Can get from LoadData
            y_probability: Probability of correctness predicted by network, shape (NUM_SLIDES, NUM_SUBJECTS). Can get from Network
        '''
        symptom_names = ['Memory_Now','Executive_Now','Language_NOW', 'Visual_Now',
                        'Motor_Now','Sensory_Now','Behavior_Now','CON_Now','Head_Now']
        NUM_SUBJECTS = len(subject_array)
        
        subject_names = []
        for subject in subject_array:
            subject_names.append(subject.Name)
        
        slide_prob_names = []
        for slide_id in range(NUM_SLIDES):
            slide_prob_names.append('Slide ' + str(slide_id+1) + ' probability')
        
        #SoP = Sum of Probabilities, PCFS = Percentage of Correct First Selections, LFS = Latency of First Selections
        SoP_names = ['SoP on easy slides', 'SoP on hard slides', 'SoP on all slides']
        PCFS_names = ['Percentage Correct on easy slides', 'Percentage Correct on hard slides', 'Percentage Correct on all slides']
        LFS_names = ['Latency on easy slides', 'Latency on hard slides', 'Latency on all slides']
        
        
        df = pd.read_csv('raw_data/Concussion_data.csv',
                        usecols = ['Subject ID','Gender','Age','YOE','# of concussions']
                        +
                        symptom_names
                        +
                        ['slf_FA_R','slf_FA_L','CC_body_FA','FAWhole','unc_FA_R','unc_FA_L',
                        'cing_FA_R','cing_FA_L'])
        for subject_id in range(len(df)):
            df.set_value(subject_id, 'Subject ID', normalize_subject_name(df['Subject ID'][subject_id]))
        df = df.set_index('Subject ID')
        
        
        df_CC = pd.read_excel('raw_data/CC& CINcomponents& SLFcomponents.xlsx')
        for subject_id in range(len(df_CC)):
            df_CC.set_value(subject_id, 'Subject ID', normalize_subject_name(df_CC['Subject ID'][subject_id]))
        df_CC = df_CC.set_index('Subject ID')
        
        
        prob_data = y_probability.T
        df_prob = pd.DataFrame(data=prob_data, index=subject_names, columns=slide_prob_names)
        
        
        SoP_data = np.empty((NUM_SUBJECTS,3))
        SoP_data[:,0] = prob_data[:,easy_slides].sum(axis=1)
        SoP_data[:,1] = prob_data[:,hard_slides].sum(axis=1)
        SoP_data[:,2] = prob_data.sum(axis=1)
        df_SoP = pd.DataFrame(data=SoP_data, index=subject_names, columns=SoP_names)
        
        
        PCFS_data = np.empty((NUM_SUBJECTS,3))
        PCFS_data[:,0] = slide_labels.T[:,easy_slides].mean(axis=1)
        PCFS_data[:,1] = slide_labels.T[:,hard_slides].mean(axis=1)
        PCFS_data[:,2] = slide_labels.T[:,range(1,NUM_SLIDES)].mean(axis=1)
        df_PCFS = pd.DataFrame(data=PCFS_data, index=subject_names, columns=PCFS_names)
        
        
        text = pd.read_csv('raw_data/mfft_sequence.txt',delimiter='\t')
        latency_arr = np.empty((NUM_SUBJECTS,NUM_SLIDES))
        for subject_id in range(NUM_SUBJECTS):
            for slide_id in range(NUM_SLIDES):
                line_start = subject_id*NUM_IMAGES*NUM_SLIDES + slide_id*NUM_IMAGES
                line_end = line_start+7
                for line in range(line_start,line_end):
                    if text.loc[line,'SelectionOrderAtt'] == 1:
                        latency_arr[subject_id,slide_id] = text.loc[line,'TimeToFirstSelectionAtt']
                        break
        LFS_data = np.empty((NUM_SUBJECTS,3))
        LFS_data[:,0] = latency_arr[:,easy_slides].mean(axis=1)
        LFS_data[:,1] = latency_arr[:,hard_slides].mean(axis=1)
        LFS_data[:,2] = latency_arr[:,range(1,NUM_SLIDES)].mean(axis=1)
        df_LFS = pd.DataFrame(data=LFS_data, index=subject_names, columns=LFS_names)
        
        
        self.df = pd.concat([df,df_CC,df_prob,df_SoP,df_PCFS,df_LFS], axis=1)
        self.df_many = self.df[self.df['# of concussions']>1] #only PCS patients with more than 1 concussion
        
        
    def output_to_csv(self):
        self.df.to_csv('create_dataframe/dataframe_all.csv')
        self.df_many.to_csv('create_dataframe/dataframe_many_concussions.csv')
