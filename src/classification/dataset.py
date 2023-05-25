from abc import abstractmethod
from light_torch.data import Dataset


class ClassificationDataset(Dataset):

    @abstractmethod
    def label_to_num(self, label: str) -> int:
        pass

    @abstractmethod
    def num_to_label(self, label: int) -> str:
        pass

    @abstractmethod
    def get_label_num(self):
        pass
