#!/usr/bin/python3
#
# By AA Aptekmann, Jul 17 2019, for ENIGMA project at YANALAB-RU
# Input/Output Function
# import parser # AA lib
import models.MeBiPred.encode as encode # AA lib 
import os
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool
from tools.parser import parse_fasta
#set size limit = None for no limit
s_lim = None

### Aux functions 
def basename(filename):
      return filename.split('/')[-1:][0].split('.')[0]

def line_to_vector(line):     
    line = line.lstrip('[').rstrip(']\n').split(', ') 
    return [ line[0].strip('\'') ] + [float(i) for i in line[1:] ]

def unpack(args):
    return encode.CTD(*args)

### Input
def encode_fasta(fasta, precoded_dict_list, kmer_dict):
    tasks = [(i.id, i.seq, precoded_dict_list, kmer_dict) for i in parse_fasta(fasta)]
    encoded = []
    if len(tasks) > 1000:
        multi = True
    else:
        multi = False    
    if multi:
        
        pool = Pool()
        encoded = pool.map(unpack, tasks)
        pool.close()
        pool.join()
    else:
        for i in tasks: 
             encoded.append( unpack(i) )     
    return encoded   

#encode with no multi-processing
def encode_sequence(sequence: SeqRecord, precoded_dict_list, kmer_dict):
    task = (sequence.id, sequence.seq, precoded_dict_list, kmer_dict)
    return [unpack(task)]

def parse_coded(coded_file_handle):
    encoded = []  
    for line in coded_file_handle:
        encoded.append(line_to_vector(line))  
    return encoded   
 
def save_encode(in_fasta, precoded_dict_list, kmer_dict, tier=''):
    cwd = os.path.dirname(__file__)
    coded_path = cwd+'/encoded_seqs/'
    b_name = basename(in_fasta)
    # code_file = coded_path + b_name + tier + '.coded' 
    #print('Saving ', code_file)    
    encoded = encode_fasta(in_fasta, precoded_dict_list, kmer_dict)
    out = open(coded_path + b_name + tier + '.coded', 'w')
    for feat_vector in encoded:
        out.write(str(feat_vector).lstrip('[').rstrip(']')+'\n')
    out.close()           
    return encoded
