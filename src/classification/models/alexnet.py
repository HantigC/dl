import torch
from torch.nn import functional as F
from torch import nn


from src.layers.lazy import LazyLinear


class AlexNet(nn.Module):
    def __init__(self, classes_num):
        super().__init__()
        self.classes_num = classes_num
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4)
        self.conv12 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4)

        self.conv21 = nn.Conv2d(
            in_channels=48, out_channels=128, kernel_size=5, padding=2
        )
        self.max_pool21 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv22 = nn.Conv2d(
            in_channels=48, out_channels=128, kernel_size=5, padding=2
        )
        self.max_pool22 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv31 = nn.Conv2d(
            in_channels=256, out_channels=192, kernel_size=3, padding=1
        )
        self.max_pool31 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv32 = nn.Conv2d(
            in_channels=256, out_channels=192, kernel_size=3, padding=1
        )
        self.max_pool32 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv41 = nn.Conv2d(
            in_channels=192, out_channels=192, kernel_size=3, padding=1
        )
        self.conv42 = nn.Conv2d(
            in_channels=192, out_channels=192, kernel_size=3, padding=1
        )

        self.conv51 = nn.Conv2d(
            in_channels=192, out_channels=128, kernel_size=3, padding=1
        )
        self.conv52 = nn.Conv2d(
            in_channels=192, out_channels=128, kernel_size=3, padding=1
        )

        self.flatten = nn.Flatten()

        self.ffn11 = LazyLinear(out_features=1024)
        self.ffn11_d = nn.Dropout()
        self.ffn12 = LazyLinear(out_features=1024)
        self.ffn12_d = nn.Dropout()

        self.ffn21 = nn.Linear(in_features=2048, out_features=1024)
        self.ffn21_d = nn.Dropout()
        self.ffn22 = nn.Linear(in_features=2048, out_features=1024)
        self.ffn22_d = nn.Dropout()

        self.ffn31 = nn.Linear(in_features=2048, out_features=classes_num)

    def forward(self, x):
        st_part = self.max_pool21(F.relu(self.conv21(F.relu(self.conv11(x)))))
        nd_part = self.max_pool22(F.relu(self.conv22(F.relu(self.conv12(x)))))

        x = torch.cat([st_part, nd_part], dim=1)
        st_part = self.conv51(
            F.relu(self.conv41(self.max_pool31(F.relu(self.conv31(x)))))
        )
        nd_part = self.conv52(
            F.relu(self.conv42(self.max_pool32(F.relu(self.conv32(x)))))
        )
        x = torch.cat([st_part, nd_part], dim=1)
        x = self.flatten(x)

        x = torch.cat(
            [
                self.ffn11_d(F.relu(self.ffn11(x))),
                self.ffn12_d(F.relu(self.ffn12(x))),
            ],
            dim=1,
        )
        x = torch.cat(
            [
                self.ffn21_d(F.relu(self.ffn21(x))),
                self.ffn22_d(F.relu(self.ffn22(x))),
            ],
            dim=1,
        )

        return self.ffn31(x)
