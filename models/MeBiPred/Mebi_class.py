import os
import numpy as np
from tf_keras.models import model_from_json
from .load_dicts import precoded_kmer_list, precoded_dict_list
from .iof import encode_fasta, encode_sequence
from .mbp import load_ANN
from Bio.SeqRecord import SeqRecord

class MeBiModel:
    def __init__(self, metals, models, mono_model, multi_model, kmer_dict, dict_list):
        self.metals = metals
        self.models = models
        self.mono_model = mono_model
        self.multi_model = multi_model
        self.kmer_dict = kmer_dict
        self. dict_list = dict_list

    @classmethod
    def build(cls, metals: list[str] = ['CA', 'CO', 'CU', 'FE', 'K', 'MG', 'MN', 'NA', 'NI', 'ZN']):
        metals = metals
        models = [load_ANN('T2'+ ion) for ion in metals ]
        multi_model = load_ANN('Multi')
        mono_model = load_ANN('Mono')
        kmer_dict = precoded_kmer_list()
        dict_list = precoded_dict_list()
        return cls(metals, models, mono_model, multi_model, kmer_dict, dict_list)

    def predict(self, input: SeqRecord) -> dict:
        task = (np.delete(encode_sequence(input, self.dict_list, self.kmer_dict), 0, 1)).astype('float')
            #predict--------------------------
        multi_pred = self.multi_model.predict(task)
        mono_pred = self.mono_model.predict(task)
        multi_pred = self.multi_model.predict(task)
        task = np.hstack((task, multi_pred, mono_pred))
        predictions = [model.predict(task) for model in self.models]
        result = {ion: round(float(predictions[j][0][0]), 2) for j, ion in enumerate(self.metals)}
        return result