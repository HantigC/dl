from collections import defaultdict
import numpy as np

from light_torch.utils.device import to_numpy_cpu
from .utils.coversion import prob_to_pred


class ClassificationMetrics:
    def __init__(self):
        self._confussion_matrix = defaultdict(lambda: defaultdict(int))
        self._inv_confussion_matrix = defaultdict(lambda: defaultdict(int))

    def add(self, gt, pred):
        self._confussion_matrix[gt][pred] += 1
        self._inv_confussion_matrix[pred][gt] += 1

    def addps(self, gts, y_pred):
        y_pred = prob_to_pred(y_pred)
        gts = to_numpy_cpu(gts)
        self.adds(gts, y_pred)

    def adds(self, gts, preds):
        for gt, pred in zip(gts, preds):
            self.add(gt, pred)

    def total(self):
        return sum(v for vs in self._confussion_matrix.values() for v in vs.values())

    def accuracy(self):
        return (
            sum(
                self._confussion_matrix[label][label]
                for label in self._confussion_matrix.keys()
            )
            / self.total()
        )

    def recall(self, label):
        def _recall(label):
            label_expected = self._confussion_matrix[label]
            return label_expected[label] / sum(label_expected.values())

        if label is not None:
            return _recall(label)
        return np.mean(_recall(label) for label in self._confussion_matrix.keys())

    def precision(self, label=None):
        def _precision(label):
            label_expected = self._inv_confussion_matrix[label]
            return label_expected[label] / sum(label_expected.values())

        if label is not None:
            return _precision(label)
        return np.mean(
            _precision(label) for label in self._inv_confussion_matrix.keys()
        )

    def fone(self, label=None, mode="macro"):
        def _fone(label):
            p = self.precision(label)
            r = self.recall(label)
            return 2 * p * r / (p + r)

        if label is not None:
            return _fone(label)
        if mode == "micro":
            np.mean(_fone(label) for label in self._confussion_matrix.keys())
        elif mode == "macro":
            p = self.precision()
            r = self.recall()
            return 2 * p * r / (p + r)
        raise ValueError(
            f"`mode` expected to be ['micro', 'macro']. Actual value: {mode}"
        )

    def get_summary(self):
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_micro": self.fone(mode="micro"),
            "f1_macro": self.fone(mode="macro"),
        }
