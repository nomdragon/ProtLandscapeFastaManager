"""
Wrapper class around MeBiPred (Metal Binding Predictor) model

Note: MeBiPred only supports scoring -- no generation, mutation scanning,
conditional scoring, or embeddings -- so this only implements Scorer.
"""
import os
from typing import Self, Sequence

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

try:
    from .mbp import load_ANN
    from .load_dicts import precoded_dict_list, precoded_kmer_list, load_aa
    from .encode import CTD
    IMPORT_AVAILABLE = True
except ImportError as e:
    print(f"MeBiPred import failed: {e}")  # add this line temporarily
    IMPORT_AVAILABLE = False


class MeBiPred(BaseModel, Scorer, ConditionalMutationScorer):
    available = IMPORT_AVAILABLE
    name: str = "MeBiPred"
    citations: list[str] = ["doi:10.1093/bioinformatics/btaa1015"]

    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = True  # independent per-sequence inference, trivially parallelizable

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    METALS = ['CA', 'CO', 'CU', 'FE', 'K', 'MG', 'MN', 'NA', 'NI', 'ZN']

    def __init__(self, ion: str | None = None, keep_model_after_pred: bool = True):
        """
        Parameters
        ----------
        ion
            If set, only load/run the single ion model (e.g. "ZN") in addition
            to the required Multi/Mono base models. If None, score all ions and
            return the max(multi, mono) metal-binding score.
        keep_model_after_pred
            If True, keep loaded models in memory between score() calls to avoid
            reloading. Set False before serializing this instance.
        """
        if not self.available:
            raise ValueError(
                "mymetal package could not be imported. Is it installed, "
                "and is tf_keras available?"
            )

        if ion is not None and ion not in self.METALS:
            raise ValueError(f"ion must be one of {self.METALS} or None, got {ion!r}")

        self.ion = ion
        self.keep_model_after_pred = keep_model_after_pred

        self._system = None
        self.dict_list = None
        self.kmer_dict_list = None
        self.valid_alphabet = None
        self.multi_model = None
        self.mono_model = None
        self.ion_models = None

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle single-component protein system"

        return True, ""

    def _load_model(self):
        if self.multi_model is not None:
            return  # already loaded

        self.dict_list = precoded_dict_list()
        self.kmer_dict_list = precoded_kmer_list()
        self.valid_alphabet = set(load_aa().keys())

        self.multi_model = load_ANN('Multi')
        self.mono_model = load_ANN('Mono')

        ions_to_load = [self.ion] if self.ion is not None else self.METALS
        self.ion_models = {ion: load_ANN('T2' + ion) for ion in ions_to_load}

    def _delete_model(self):
        self.multi_model = None
        self.mono_model = None
        self.ion_models = None

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    def _validate_alphabet(self, seq: str) -> None:
        bad_chars = set(seq) - self.valid_alphabet
        if bad_chars:
            raise ValueError(
                f"Sequence contains characters not in MeBiPred's amino acid "
                f"alphabet: {sorted(bad_chars)}"
            )

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()
        self._validate_instances(instances)

        sequences = ["".join(instance[0].rep) for instance in instances]

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            for seq in sequences:
                self._validate_alphabet(seq)

            scores = []
            for i, seq in enumerate(sequences):
                if status_callback:
                    status_callback(
                        "running", (i / len(sequences)) * 100,
                        f"Scoring sequence {i + 1}/{len(sequences)}"
                    )

                try:
                    row = CTD("query", seq, self.dict_list, self.kmer_dict_list)
                except Exception as e:
                    raise RuntimeError(
                        f"MeBiPred feature encoding failed for sequence {i}: {e}"
                    ) from e

                features = np.array(row[1:], dtype=float).reshape(1, -1)

                try:
                    c = self.multi_model(features, training=False).numpy()
                    d = self.mono_model(features, training=False).numpy()
                except Exception as e:
                    raise RuntimeError(
                        f"MeBiPred base model prediction failed for sequence {i} "
                        f"(feature vector length {features.shape[1]}): {e}"
                    ) from e

                stacked = np.hstack((features, c, d))

                if self.ion is not None:
                    try:
                        p = self.ion_models[self.ion](stacked, training=False).numpy()
                    except Exception as e:
                        raise RuntimeError(
                            f"MeBiPred ion model ({self.ion}) prediction failed "
                            f"for sequence {i}: {e}"
                        ) from e
                    scores.append(float(p[0][0]))
                else:
                    scores.append(float(max(c[0][0], d[0][0])))

        return np.array(scores)