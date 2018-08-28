import numpy as np

seed = 7
np.random.seed(seed)
NUM_SLIDES = 21

def modify_sequences_in_slide(sequences_in_slide, num_last_arg, mode, section, num_first_arg=None):
    '''
    sequences_in_slide: list of NUM_SUBJECTS elements, each is the sequence of that subject
    num_last: number of fixations at the end of sequences that we care about. Can be an integer, or 'half' i.e. half of the sequence
    mode: 'shuffle' or 'discard'
    section: the section of the sequence we're modifying. 'pre' or 'post'
    
    e.g. num_last_arg=5, mode='discard', section='pre' -> discard everything except last 5
         num_last_arg='half', mode='shuffle', section='post' -> keep 1st half the same, shuffle second half 
    '''
    NUM_SUBJECTS = len(sequences_in_slide)
    #deep copy
    ret_seqs = np.copy(sequences_in_slide)
    for subject_id in range(NUM_SUBJECTS):
        ret_seqs[subject_id] = np.copy(sequences_in_slide[subject_id])
    
    #modify sequences
    for subject_id in range(NUM_SUBJECTS):
        if num_first_arg != None and mode=='keep' and section=='pre':
            if isinstance(num_first_arg, int):  #absolute number of fixations to keep
                num_first_keep = num_first_arg
            elif isinstance(num_first_arg, float):      #ratio of sequence to keep
                num_first_keep = int(num_first_arg*len(ret_seqs[subject_id]))
            else:
                print('Error: num_first_arg has to be a number')
                break
            ret_seqs[subject_id] = ret_seqs[subject_id][:num_first_keep]
        elif mode == 'discard last dup':
            if len(ret_seqs[subject_id]) > 0:
                last_fix = ret_seqs[subject_id][-1]
                num_last_dup = 0
                while num_last_dup < len(ret_seqs[subject_id]) and ret_seqs[subject_id][-num_last_dup-1] == last_fix:
                    num_last_dup += 1
                ret_seqs[subject_id] = ret_seqs[subject_id][:-num_last_dup]
        else:
            if num_last_arg == 'half':
                num_last = int(len(ret_seqs[subject_id])/2)
            else:
                num_last = num_last_arg
            if mode=='shuffle' and section=='pre':
                np.random.shuffle(ret_seqs[subject_id][:-num_last])
            elif mode=='shuffle' and section=='post':
                np.random.shuffle(ret_seqs[subject_id][-num_last:])
            elif mode=='discard' and section=='pre':
                ret_seqs[subject_id] = ret_seqs[subject_id][-num_last:]
            elif mode=='discard' and section=='post':
                ret_seqs[subject_id] = ret_seqs[subject_id][:-num_last] if num_last > 0 else ret_seqs[subject_id]
            else:
                print('Error: Invalid argument(s)')
                break
    return ret_seqs


def modify_all_sequences(all_sequences, num_last_arg, mode, section):
    '''
    Arguments:
        all_sequences: list of 21 elements, each is the list of sequences of each subject for a particular slide.
            (each element is a sequences_in_slide. FMI, see Parameters in modify_sequences_in_slide above)
    '''
    ablated_sequences = np.copy(all_sequences)
    for slide_id in range(NUM_SLIDES):
        print('Modify sequence in slide ', slide_id+1)
        ablated_sequences[slide_id] = modify_sequences_in_slide(all_sequences[slide_id], num_last_arg, mode, section)
    return ablated_sequences
    
