#!/usr/bin/python
# A Aptekmann imp of MBP1 for ENIGMA at RU
# May 2019
# Library to encode SeqRecords into features for machine learning algorithms
import random, sys, os
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
# from inspect import signature

def mkdir(dir):
    if os.path.isdir(dir):
        return 0
    else :
        os.mkdir(dir)    

def format_id(id):
    return id.split('|')[0].replace(':','_').upper()

# Condensed AA composition
def C3_feats(coded_seq):
    C1 = float(coded_seq.count('L')) / len(coded_seq)
    C2 = float(coded_seq.count('M')) / len(coded_seq)
    C3 = float(coded_seq.count('H')) / len(coded_seq)
    return [C1, C2, C3]

# AA composition
def C20_feats(seq, precoded_dict):
    largo = float( len(seq)) 
    feats = [ seq.count(i) / largo  for i in precoded_dict.keys() ]
    return feats

def T3_feats(coded_seq):
    # 3 posible transitions L <-> H ; L <-> M ; M <-> H
    LH, LM, MH = 0, 0, 0
    # Reduce chars typed
    a = coded_seq
    # At max n-1 transitions
    l1 = len(coded_seq) - 1
    if l1 == 0:
        return [0.0,0.0,0.0]
    for i in range(l1):
        if a[i] == 'L':
            if a[i+1] == 'M':    # L -> M
                LM += 1
            elif a[i+1] == 'H':  # L -> H
                LH += 1
        elif a[i] == 'M':
            if a[i+1] == 'L':    # M -> L
                LM += 1
            elif a[i+1] == 'H':  # M -> H
                MH += 1
        elif a[i] == 'H':
            if a[i+1] == 'M':    # H -> M
                MH += 1
            elif a[i+1] == 'L':  # H -> L
                LH += 1
    LH, LM, MH = float(LH) / l1, float(LM) / l1, float(MH) / l1
    return [LH, LM, MH]

# Encode the distribution of an AA property
# Returns the position in sequence (expressed as a percent of sequence length)
# For the first ocurrence, 25%, 50%, 75, and last of a property
def D5_feats(coded_seq, K):
    count = coded_seq.count(K)
    if count == 0:
        return [0, 0, 0, 0, 0]
    l1 = float(len(coded_seq))
    # if K is not in ['L','M','H'] wont work
    K000 = coded_seq.index(K) / l1
    K025 = coded_seq.index(K, int(count * 0.25)) / l1
    K050 = coded_seq.index(K, int(count * 0.50)) / l1
    K075 = coded_seq.index(K, int(count * 0.75)) / l1
    K100 = coded_seq.index(K, int(count - 1)) / l1
    return [K000, K025, K050, K075, K100]

# Calls D5_feats for each class of AA (Low,Medium,High)
def D15_feats(coded_seq):
    DL = D5_feats(coded_seq, 'L')
    DM = D5_feats(coded_seq, 'M')
    DH = D5_feats(coded_seq, 'H')
    return DL + DM + DH


def CTD21_feats(seq, clustered_dict):
    coded_seq = [clustered_dict[i] for i in seq]
    C3 = C3_feats(coded_seq)
    T3 = T3_feats(coded_seq)
    D15 = D15_feats(coded_seq)
    return C3 + T3 + D15

def KMER_feat(seq, kmer_dict): 
    win_len = 5
    windows = break_into_windows(seq, win_len)
    Kscore = 0 
    for w in windows:
        if w in kmer_dict.keys():
            Kscore += kmer_dict[w]                    
    return [Kscore]

# This is called ONCE per record, any repetitive precalculation should be made before!    
def CTD(id, seq, precoded_dict_list, kmer_dict_list):
    # Have the record id as the first feat
    feats = [format_id(id) ]
    # Then add the CTD feats
    for clustered_dict in precoded_dict_list:
        feats = feats + CTD21_feats(seq, clustered_dict)
    # Then add composition feat    
    feats = feats + C20_feats(seq, precoded_dict_list[0])    
    # Then add kmer feat
    for kmer_dict in kmer_dict_list:
        feats = feats + KMER_feat(seq, kmer_dict)    
    return feats


# # Deprecate functions kept because i bet ill need them.
def break_into_windows(seq, win_len):
    windows = [seq[i:i+win_len] for i in range(len(seq)-win_len-1)]
    return windows

# def remove_label(encoded_data):
#     return [ i[1:] for i in encoded_data ]

# def encode(seq_string, code_d):
#     encoded = [code_d[char] for char in seq_string]
#     return encoded

# def fasta_to_coded(seq_record_list, code_d, win_len):
#     # Parse fasta to Seq_records
#     print('Read %s sequences' % len(seq_record_list))
#     # Break input Seq_records into windows
#     seq_window_list = []
#     for windows in [break_into_windows(seq_rec, win_len)
#                     for seq_rec in seq_record_list]:
#         for window in windows:
#             seq_window_list.append(window)
#     # Encode windows
#     encoded = [encode(window, code_d) for window in seq_window_list]
#     return encoded
