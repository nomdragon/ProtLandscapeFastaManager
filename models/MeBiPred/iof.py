#!/usr/bin/python3
#
# By AA Aptekmann, Jul 17 2019, for ENIGMA project at YANALAB-RU
# Input/Output Functions
from Bio import SeqIO
# import parser # AA lib
import encode as encode # AA lib 
import os
from multiprocessing import Pool
#set size limit = None for no limit
s_lim = None

### Aux functions 
def basename(filename):
      return filename.split('/')[-1:][0].split('.')[0]

def line_to_vector(line):     
    line = line.lstrip('[').rstrip(']\n').split(', ') 
    return [ line[0].strip('\'') ] + [float(i) for i in line[1:] ]

def parse_fasta(fasta_filename):
    seq_record_list = [rec for rec in SeqIO.parse(fasta_filename, "fasta")]
    return seq_record_list


def unpack(args):
    return encode.CTD(*args)

### Input
def encode_fasta(fasta, precoded_dict_list, kmer_dict):
    tasks = [(i.id, i.seq, precoded_dict_list, kmer_dict) for i in parse_fasta(fasta)[0: s_lim] ]
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

# def check_consistency(fasta, coded_file_handle, precoded_dict_list, kmer_dict, tier=''):
#     records = parser.parse_fasta(fasta)
#     c_vector = encode.CTD(records[0], precoded_dict_list, kmer_dict)
#     #print (c_vector)
#     lines  = [line for line in coded_file_handle]
#     if len(lines) == 0:
#         print(coded_file_handle, 'is an empty file.')
#         return False
#     first_line = line_to_vector(lines[0])
#     coded_file_handle.close()
#     if c_vector == first_line and len(records) == len(lines):
#         #print('Consistency')
#         return True   
#     else:
#         print(c_vector,'\n', first_line )
#         print('No consistency')
#         return False    

def parse_coded(coded_file_handle):
    encoded = []  
    for line in coded_file_handle:
        encoded.append(line_to_vector(line))  
    return encoded   

def load_encode(in_fasta, precoded_dict_list, kmer_dict, tier=''):
    encoded = encode_fasta(in_fasta, precoded_dict_list, kmer_dict)
    return encoded
    # coment above and uncoment below to save pre coded fastas
    # cwd = os.path.dirname(__file__)
    # coded_path = cwd+'/encoded_seqs/'
    # b_name = basename(in_fasta)
    # code_file = coded_path + b_name + tier + '.coded' 
    # try:
    #     coded_file_handle = open(code_file, 'r') 
    #     #print(code_file , 'Found, checking coding consistency.')
    #     if check_consistency(in_fasta, coded_file_handle, precoded_dict_list, kmer_dict):
    #         coded_file_handle.close()
    #         coded_file_handle = open(code_file, 'r') 
    #         return parse_coded(coded_file_handle)
    #     else :
    #         encoded = save_encode(in_fasta, precoded_dict_list, kmer_dict, tier)
    #         return encoded           
    # except FileNotFoundError:
    #       #print(coded_path + b_name + '.coded', 'Not found, will code it from start.')
    #       coded_file_handle = open(code_file, 'w')
    #       coded_file_handle.close()
    #       encoded = save_encode(in_fasta, precoded_dict_list, kmer_dict, tier)
    #       return encoded
## Output
def save_out(line_list,name):
    out = open(name,'w')
    for item in line_list:
        out.write( str(item) +'\n')
    out.close()   

def save_out_csv(line_list,name):
    out = open(name,'w')
    for item in line_list:
        out.write( str(item) +'\n')
    out.close()   

def save_csv(line_list,name):
    out = open(name,'w')
    for item in line_list:
        for subitem in item:
            out.write(str(subitem) + '\t')
        out.write('\n')        
    out.close()   


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
