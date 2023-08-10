import torch.nn as nn
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights

class Net(nn.Module):
    """
    Class implementing the transfer learning model using AlexNet.
    """
    def __init__(self, img_dim):
        super().__init__()

        # Instatiate model with pretrained weights
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)

        for param in self.model.parameters():
             param.requires_grad = False


        # Now we adapt the AlexNet classifiers
        self.model.classifier[1] = nn.Linear(9216, 1024)
        self.model.classifier[4] = nn.Linear(1024, 256)

        self.model.features.add_module("7_bn", nn.BatchNorm2d(256))
        self.model.features.add_module("8_dropout", nn.Dropout(0.5))
        self.model.features.add_module("9_relu", nn.ReLU(inplace=True))

        # Updating the third and last classifier that is the output layer of the network.
        self.model.classifier[6] = nn.Linear(256, 64)

        # Additional classifiers to improve models performance
        self.model.classifier.add_module("7_dropout", nn.Dropout(0.5))
        self.model.classifier.add_module("8_relu", nn.ReLU(inplace=True))
        self.model.classifier.add_module("9_linear", nn.Linear(64, 32))
        self.model.classifier.add_module("10_relu", nn.ReLU(inplace=True))
        self.model.classifier.add_module("11_linear", nn.Linear(32, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        """
        Forward pass.
        Args:
            img:
                Images to calculate the forward pass.
        """
        img = self.model.forward(img)
        img = self.sigmoid(img)
        img = img.flatten()
        return img