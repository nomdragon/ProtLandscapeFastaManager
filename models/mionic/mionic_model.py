import torch
from pathlib import Path
from .model.models import IonicProtein
import torch.nn as nn
from Bio.SeqRecord import SeqRecord

def build_mionic() -> tuple:
    #Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #@title Load ESM-2 model and extract per-residue representations

    # Load ESM-2 model
    model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval()  # disables dropout for deterministic results
    node_dim = 1280
    model_ionic = IonicProtein(node_dim)
    weights_path = Path(__file__).resolve().parent / 'esm2_t33_650M_UR50D_setB_fold1.pt'
    state_dict = torch.load(weights_path, map_location=device)
    model_ionic.load_state_dict(state_dict, strict=False)
    model_ionic.to(device).eval()
    m = nn.Sigmoid()
    return (model_esm, alphabet, batch_converter, model_ionic, device, m)

#TODO implement custom ion parameter
def mionic_predict(input: SeqRecord, mionic: tuple, target_prob: float) -> dict:
    #Initialize model
    model_esm=mionic[0]
    alphabet=mionic[1]
    batch_converter=mionic[2]
    model_ionic=mionic[3]
    device=mionic[4]
    m=mionic[5]
    jobname=input.id
    sequence_str=input.seq

    data = [(jobname, sequence_str)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    sequence_representations = (token_representations[0, 1 : len(sequence_str) + 1])

    fields = ['CA','CO','CU','FE2','FE','MG','MN','PO4','SO4','ZN', 'null']

    ion_predictions = {}
    with torch.no_grad():
        emb1 = torch.squeeze(sequence_representations)
        outputs = model_ionic(emb1.to(device), mask=None)
        predictions_sigmoid = m(torch.stack(outputs))

        for ind, ion in enumerate(fields):
            if ion == 'null':
                continue
            pred_ion = predictions_sigmoid[ind].cpu().detach().numpy()
            hits = {i + 1: round(val.item(), 2) for i, val in enumerate(pred_ion) if val > target_prob}
            if hits:
                ion_predictions[ion] = hits

    return ion_predictions

