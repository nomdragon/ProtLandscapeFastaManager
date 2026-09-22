import torch
import pandas as pd
import re
from pathlib import Path
from package.training.models import *
import torch.nn as nn
import os
import time



query_sequence = 'MFENITAAPADPILGLADLFRADERPGKINLGIGVYKDETGKTPVLTSVKKAEQYLLENETTKNYLGIDGIPEFGRCTQELLFGKGSALINDKRARTAQTPGGTGALRVAADFLAKNTSVKRVWVSNPSWPNHKSVFNSAGLEVREYAYYDAENHTLDFDALINSLNEAQAGDVVLFHGCCHNPTGIDPTLEQWQTLAQLSVEKGWLPLFDFAYQGFARGLEEDAEGLRAFAAMHKELIVASSYSKNFGLYNERVGACTLVAADSETVDRAFSQMKAAIRANYSNPPAHGASVVATILSNDALRAIWEQELTDMRQRIQRMRQLFVNTLQEKGANRDFSFIIKQNGMFSFSGLTKEQVLRLREEFGVYAVASGRVNVAGMTPDNMAPLCEAIVAVL'
jobname = 'P0ACS5'
data = [
    (jobname, query_sequence)]
m = nn.Sigmoid()
node_dim=1280
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#@title Load ESM-2 model and extract per-residue representations

# Load ESM-2 model
model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
start_time = time.perf_counter()
batch_converter = alphabet.get_batch_converter()
model_esm.eval()  # disables dropout for deterministic results

model_ionic = IonicProtein(node_dim)
state_dict = torch.load('esm2_t33_650M_UR50D_setB_fold1.pt', map_location=device)
model_ionic.load_state_dict(state_dict, strict=False)
model_ionic.to(device).eval()

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
#print (batch_labels, len(batch_strs[0]), len(batch_tokens[0]), batch_lens)

with torch.no_grad():
    results = model_esm(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

sequence_representations = (token_representations[0, 1 : len(query_sequence) + 1])
#token_representations.shape, sequence_representations.shape

#@title Display predictions

fields = ['CA','CO','CU','FE2','FE','MG','MN','PO4','SO4','ZN', 'null']

label_dict = dict.fromkeys(fields, [])

ion_residues = {}
ion_probabilities = {}
with torch.no_grad():
    emb1 = torch.squeeze(sequence_representations)
    outputs = model_ionic(emb1.to(device), mask=None)
    predictions_sigmoid = m(torch.stack(outputs))
    predictions_binary = torch.round(predictions_sigmoid)

    for ind, ion in enumerate(fields):
        if ion == 'null':
            continue
        pred_ion = predictions_sigmoid[ind].cpu().detach().numpy()
        positions = [i + 1 for i, val in enumerate(pred_ion) if val > 0.8]
        probabilities = [round(val.item(), 2) for i, val in enumerate(pred_ion) if val > 0.8]
        if positions:
            ion_residues[ion] = positions
            ion_probabilities[ion] = probabilities

print(f'Hits: {ion_residues}')
print(f'Confidence: {ion_probabilities}')


