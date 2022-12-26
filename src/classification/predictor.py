import torch


class ClassificationPredictor:
    def __init__(self, model, class_maper=None):
        self._model = model
        self._class_mapper = class_maper

    def __call__(self, inpt):
        probabilities = self._model(inpt)
        labels = torch.argmax(probabilities, dim=1).cpu().numpy()
        if self._class_mapper is not None:
            labels = [self._class_mapper.id2name(idd) for idd in labels]

        return labels
