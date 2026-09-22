from tools.parser import *
from models.MeBiPred.mbp import mbp_predict_one, load_MeBiPred_models
from models.mionic.mionic_model import build_mionic, mionic_predict
#save CSV to fasta with any column num
"""
csv_to_fasta("exampledatasets/mbpa_org_seq_uniprot_bindingsite(in).csv", 
             "uniprot_binding_set2.fasta",
              id_col=2,
              seq_col=1, 
              info_col=0, 
              first_row=30, 
              n_rows=20)
binding_sequences = parse_fasta("exampledatasets/uniprot_bindingsite.fasta")
"""
#MeBiPred predict fasta
binding_sequences = parse_fasta("exampledatasets/uniprot_bindingsite.fasta")
MeBiPred = load_MeBiPred_models()
Mionic = build_mionic()
for sequence in binding_sequences:
    print(f'{sequence.id}: {mbp_predict_one(sequence, MeBiPred)}')
    print(f'{sequence.id}: {mionic_predict(sequence, mionic=Mionic, target_prob=0.5)}')
    
