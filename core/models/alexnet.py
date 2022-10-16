import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        """
        URL: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
        @article{krizhevsky2017imagenet,
          title={Imagenet classification with deep convolutional neural networks},
          author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
          journal={Communications of the ACM},
          volume={60},
          number={6},
          pages={84--90},
          year={2017},
          publisher={AcM New York, NY, USA}
        }
        """
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
                               stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,
                               stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,
                               stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=384)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,
                               stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=384)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,
                               stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x):
        """

        :param x: torch.Tensor, shape: [batch_size, 3, 227, 227]
        :return:
        """
        x = self.bn1(self.conv1(x))   # (batch, 96, 55, 55)
        x = self.pool1(x)             # (batch, 96, 27, 27)

        x = self.bn2(self.conv2(x))   # (batch, 256, 27, 27)
        x = self.pool2(x)             # (batch, 256, 13, 13)

        x = self.bn3(self.conv3(x))    # (batch, 384, 13, 13)
        x = self.bn4(self.conv4(x))    # (batch, 384, 13, 13)
        x = self.bn5(self.conv5(x))    # (batch, 256, 13, 13)
        x = self.pool3(x)              # (batch, 256, 6, 6)

        x = self.flatten(x)    # (batch, 9216)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.activation(self.fc3(x))

        return x

    @staticmethod
    def get_input_shape():
        return 227, 227

    def __repr__(self):
        return "AlexNet"


if __name__ == '__main__':
    sample_x = torch.rand(1, 3, 227, 227)     # N, C, H, W
    model = AlexNet(num_classes=10)
    y = model(sample_x)
    print(y.size())
    print(y)

