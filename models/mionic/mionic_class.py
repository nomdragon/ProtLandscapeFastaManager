#TODO: implement model class
import torch
from pathlib import Path
from .model.models import IonicProtein
import torch.nn as nn
from Bio.SeqRecord import SeqRecord

class MionicModel:
    def __init__(self, model_esm, alphabet, batch_converter, model_ionic, device):
        self.model_esm = model_esm
        self.alphabet = alphabet
        self.batch_converter = batch_converter
        self.model_ionic = model_ionic
        self.device = device
        self.m = nn.Sigmoid()

    @classmethod
    def build(cls, weights_path=Path(__file__).resolve().parent / 'esm2_t33_650M_UR50D_setB_fold1.pt', device=None):
        resolved_device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        model_esm.eval()
        batch_converter = alphabet.get_batch_converter()

        model_ionic = IonicProtein(feature_dim=1280)
        state_dict = torch.load(weights_path, map_location=resolved_device)
        model_ionic.load_state_dict(state_dict, strict=False)
        model_ionic.to(resolved_device).eval()
        model_esm.to(resolved_device)

        return cls(model_esm, alphabet, batch_converter, model_ionic, resolved_device)

    def predict(self, input: SeqRecord, target_prob: float = 0.8) -> dict:
        jobname = input.id
        sequence_str = input.seq
        data = [(jobname, sequence_str)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        with torch.no_grad():
            results = self.model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        sequence_representations = token_representations[0, 1: len(sequence_str) + 1]

        fields = ['CA', 'CO', 'CU', 'FE2', 'FE', 'MG', 'MN', 'PO4', 'SO4', 'ZN', 'null']
        ion_predictions = {}
        with torch.no_grad():
            emb1 = torch.squeeze(sequence_representations)
            outputs = self.model_ionic(emb1.to(self.device), mask=None)
            predictions_sigmoid = self.m(torch.stack(outputs))

            for ind, ion in enumerate(fields):
                if ion == 'null':
                    continue
                pred_ion = predictions_sigmoid[ind].cpu().detach().numpy()
                hits = {i + 1: round(val.item(), 2) for i, val in enumerate(pred_ion) if val > target_prob}
                if hits:
                    ion_predictions[ion] = hits

        return ion_predictions