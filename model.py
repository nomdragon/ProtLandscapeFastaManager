from abc import ABC, abstractmethod

#Abstract class for scoring models
class ScorerModel(ABC):
    @abstractmethod
    def predict(self, sequence: str):
        pass