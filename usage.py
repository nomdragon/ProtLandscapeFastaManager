from tools.parser import *
from models.mionic.mionic_model import build_mionic, mionic_predict
from models.MeBiPred.Mebi_class import MeBiModel
from models.mionic.mionic_class import MionicModel
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
Mebi = MeBiModel.build()
Mionic = MionicModel.build()
for sequence in binding_sequences:
    print(f'{sequence.id}: {Mebi.predict(sequence)}')
    print(f'{sequence.id}: {Mionic.predict(sequence, 0.5)}')
    
