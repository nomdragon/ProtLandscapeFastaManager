#!/usr/bin/python
import sys, os
import numpy as np
from tf_keras.models import model_from_json
from .load_dicts import precoded_kmer_list, precoded_dict_list
from .iof import encode_fasta, encode_sequence
from Bio.SeqRecord import SeqRecord

TF_ENABLE_ONEDNN_OPTS=0
# Standalone version just to predict 

def load_ANN(attribute):
    #encode.mkdir('./ModelPersistency/')
    cwd = os.path.dirname(os.path.abspath(__file__))
    #print(cwd)
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

#TODO: Take flags for custom ion list 
def load_MeBiPred_models() -> tuple:
    metals =  ['CA', 'CO', 'CU', 'FE', 'K', 'MG', 'MN', 'NA', 'NI', 'ZN']
    models = [load_ANN('T2'+ ion) for ion in metals ]
    multi_model = load_ANN('Multi')
    mono_model = load_ANN('Mono')
    kmer_dict = precoded_kmer_list()
    dict_list = precoded_dict_list()
    return (models, mono_model, multi_model, kmer_dict, dict_list)

#TODO: See above
def mbp_predict_one(sequence: SeqRecord, MeBiPred):
    #TODO map dict to metal ion chebi codes. current metals represent all oxidation states
    #receive model components
    models = MeBiPred[0]
    mono_model = MeBiPred[1]
    multi_model = MeBiPred[2]
    kmer_dict = MeBiPred[3]
    dict_list = MeBiPred[4]
    #build task-----------------------------
    task = (np.delete(encode_sequence(sequence, dict_list, kmer_dict), 0, 1)).astype('float')
    #predict--------------------------
    multi_pred = multi_model.predict(task)
    mono_pred = mono_model.predict(task)
    multi_pred = multi_model.predict(task)
    task = np.hstack((task, multi_pred, mono_pred))
    predictions = [model.predict(task) for model in models]
    metals =  ['CA', 'CO', 'CU', 'FE', 'K', 'MG', 'MN', 'NA', 'NI', 'ZN']
    result = {ion: round(float(predictions[j][0][0]), 2) for j, ion in enumerate(metals)}
    return result


