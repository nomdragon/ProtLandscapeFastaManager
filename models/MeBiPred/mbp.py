#!/usr/bin/python
import sys, os
import numpy as np
from tf_keras.models import model_from_json
from .load_dicts import precoded_kmer_list, precoded_dict_list
from .iof import encode_fasta, encode_sequence
from Bio.SeqRecord import SeqRecord

def load_ANN(attribute):
    #encode.mkdir('./ModelPersistency/')
    cwd = os.path.dirname(os.path.abspath(__file__))
    fname = cwd+'/ModelPersistency/ANNmodel' + str(attribute) + '.json'
    json_file = open(fname, 'r')
    # Load model architechture
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(
        cwd+'/ModelPersistency/ANNmodel' +
        str(attribute) +
        ".h5")
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])
    #print("Loaded model %s from disk" % attribute)
    json_file.close()
    return loaded_model
