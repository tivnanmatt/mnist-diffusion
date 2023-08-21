import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from diffusion import ScalarDiffusionModel_VariancePreserving_LinearSchedule

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_list = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_list)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_list)

mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)


class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.batchnorm1 = torch.nn.BatchNorm2d(out_channels)
        self.batchnorm2 = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.batchnorm1(self.conv1(x)))
        x = torch.nn.functional.relu(self.batchnorm2(self.conv2(x)))
        return x
    
class FullyConnectedBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, out_channels)
        self.batchnorm1 = torch.nn.BatchNorm1d(out_channels)
        self.batchnorm2 = torch.nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.batchnorm1(self.fc1(x)))
        x = torch.nn.functional.relu(self.batchnorm2(self.fc2(x)))
        return x

class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_base_filters=32):
        super().__init__()

        self.image_size = 28

        self.conv1 = ConvolutionalBlock(in_channels, num_base_filters)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvolutionalBlock(num_base_filters, num_base_filters*2)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvolutionalBlock(num_base_filters*2, num_base_filters*4)
        self.flatten6 = torch.nn.Flatten()
        self.fc7 = FullyConnectedBlock((self.image_size//4)*(self.image_size//4)*num_base_filters*4, 1024)
        self.fc8 = FullyConnectedBlock(1024, (self.image_size//4)*(self.image_size//4)*num_base_filters*4)
        self.unflatten9 = torch.nn.Unflatten(1, (num_base_filters*4, self.image_size//4, self.image_size//4))
        self.conv10 = ConvolutionalBlock(num_base_filters*8, num_base_filters*4)
        self.upconv11 = torch.nn.ConvTranspose2d(num_base_filters*4, num_base_filters*2, kernel_size=2, stride=2)
        self.conv12 = ConvolutionalBlock(num_base_filters*4, num_base_filters*2)
        self.upconv13 = torch.nn.ConvTranspose2d(num_base_filters*2, num_base_filters, kernel_size=2, stride=2)
        self.conv14 = ConvolutionalBlock(num_base_filters*2, num_base_filters)
        self.conv15 = torch.nn.Conv2d(num_base_filters, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool2(x1)
        x3 = self.conv3(x2)
        x4 = self.pool4(x3)
        x5 = self.conv5(x4)
        x6 = self.flatten6(x5)
        x7 = self.fc7(x6)
        x8 = self.fc8(x7)
        x9 = self.unflatten9(x8)
        x10 = self.conv10(torch.cat((x9, x5), dim=1))
        x11 = self.upconv11(x10)
        x12 = self.conv12(torch.cat((x11, x3), dim=1))
        x13 = self.upconv13(x12)
        x14 = self.conv14(torch.cat((x13, x1), dim=1))
        x15 = self.conv15(x14)
        return x15


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels=32, expandToImage=True):
        super().__init__()
        self.fc1 = FullyConnectedBlock(1, 256)
        self.fc2 = FullyConnectedBlock(256, 256)
        self.fc3 = FullyConnectedBlock(256, out_channels)
        self.expandToImage = expandToImage
    def forward(self, t):
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        x = self.fc1(t)
        x = self.fc2(x)
        x = self.fc3(x)
        if self.expandToImage:
            t_enc = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 28, 28)
        else:
            t_enc = x
        return t_enc

class PositionEncoder(torch.nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        
        xGrid, yGrid = torch.meshgrid(torch.linspace(-1, 1, 28), torch.linspace(-1, 1, 28))
        xGrid = xGrid.unsqueeze(0).unsqueeze(0)
        yGrid = yGrid.unsqueeze(0).unsqueeze(0)
        rGrid = torch.sqrt(xGrid**2 + yGrid**2)
        self.xyGrid = torch.cat((xGrid, yGrid,rGrid), dim=1).to(device)

        self.conv1 = ConvolutionalBlock(3, 32)
        self.conv2 = ConvolutionalBlock(32, 32)
        self.conv3 = ConvolutionalBlock(32, out_channels)
    
    def forward(self):
        x = self.conv1(self.xyGrid)
        x = self.conv2(x)
        pos_enc = self.conv3(x)
        return pos_enc
        

class BayesianClassifier(torch.nn.Module):
    def __init__(self, time_encoder, input_channels, output_channels):
        super().__init__()
        self.time_encoder = time_encoder
        self.conv1 = ConvolutionalBlock(input_channels, 32)
        self.conv2 = ConvolutionalBlock(32, 32)
        self.conv3 = ConvolutionalBlock(32, 32)
        self.fc1 = FullyConnectedBlock(32*28*28, 512)
        self.fc2 = FullyConnectedBlock(512, 256)
        self.fc3 = FullyConnectedBlock(256, output_channels)
        

    def forward(self, x_t, t):
        t_enc = self.time_encoder(t)
        x = torch.concat((x_t, t_enc), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 32*28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x =  torch.nn.functional.softmax(x, dim=1)
        x = x + 1e-4  # To avoid log(0)
        x = x / x.sum(dim=1, keepdim=True)
        return x