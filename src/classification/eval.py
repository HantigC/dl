from collections import defaultdict


class ClassificationMetrics:
    def __init__(self):
        self._confussion_matrix = defaultdict(lambda: defaultdict(int))

    def add(self, gt, pred):
        self._confussion_matrix[gt][pred] += 1

    def adds(self, gts, preds):
        for gt, pred in zip(gts, preds):
            self.add(gt, pred)

    @property
    def total(self):
        return sum(v for vs in self._confussion_matrix.values() for v in vs.values())

    @property
    def accuracy(self):
        return (
            sum(
                self._confussion_matrix[label][label]
                for label in self._confussion_matrix.keys()
            )
            / self.total
        )

    def get_summary(self):
        return {"accuracy": self.accuracy}
