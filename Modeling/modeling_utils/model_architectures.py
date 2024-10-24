import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class custom_CNN2d(nn.Module):
    def __init__(self, out_dim):
        super(custom_CNN2d, self).__init__()
        self.out_dim = out_dim
        self.cnn1 = nn.LazyConv2d(512, (7, 7), padding=3)
        # self.maxpool1 = nn.MaxPool2d(2)
        # self.cnn2 = nn.LazyConv2d(128, (5, 5), padding=2)
        self.l1 = nn.LazyLinear(512)
        self.l2 = nn.LazyLinear(256)
        self.skip_projection = nn.LazyLinear(256)
        self.l3 = nn.LazyLinear(128)
        self.lout = nn.LazyLinear(out_dim)
        
        #dropout layers
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        self.drop4 = nn.Dropout(0.2)

    def forward(self, x):
        x_skip = x.clone().flatten(1)
        x = F.relu(self.cnn1(x))
        x = self.maxpool1(x)
        x = F.relu(self.cnn2(x))
        x = x.flatten(1)
        x = self.drop1(x)
        x = F.relu(self.l1(x))
        x = self.drop2(x)
        x = F.relu(self.l2(x)) + self.skip_projection(x_skip)  #skips from input
        x = self.drop3(x)
        x = F.relu(self.l3(x))
        x = self.drop4(x)
        x = F.sigmoid(self.lout(x))
        return x

class FullCNN_1d(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(FullCNN_1d, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.output = nn.LazyLinear(self.out_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.flatten(1)
        x = self.output(x)
        return x

class FullLinear(nn.Module):
    def __init__(self, out_dim):
        super(FullLinear, self).__init__()
        self.out_dim = out_dim
        self.l1 = nn.LazyLinear(2056)
        self.l2 = nn.LazyLinear(1028)
        self.l3 = nn.LazyLinear(512)
        self.l4 = nn.LazyLinear(256)
        self.l5 = nn.LazyLinear(128)
        self.lout = nn.LazyLinear(out_dim)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = x.flatten(1)
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.lout(x)
        
    
class hybrid_block_1d(nn.Module):
    def __init__(self, out_dim, internal_dim=512):
        super(hybrid_block_1d, self).__init__()
        self.out_dim = out_dim
        self.cnn_layer = nn.LazyConv1d(internal_dim, 7, padding=3)
        self.linear_layer = nn.LazyLinear(internal_dim)
        self.output_layer = nn.LazyLinear(out_dim)
    
    def forward(self, x):
        x_cnn = F.relu(self.cnn_layer(x))
        x_cnn = x_cnn.flatten(1)
        x_linear = F.relu(self.linear_layer(x))
        x_linear = x_linear.flatten(1)
        x = torch.concat((x_cnn, x_linear), dim=1)
        x = self.output_layer(x)
        return x
        
    
class hybridNet(nn.Module):
    def __init__(self, num_blocks, out_dim):
        super(hybridNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.blocks = nn.ModuleList([hybrid_block_1d(512) for _ in range(num_blocks)])
        self.output = nn.LazyLinear(out_dim)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.output(x)
        return x
    
def get_resnet(input_chans=7, output_size=1, pretrained=True):
    # Load the pretrained model
    model = models.resnet18(pretrained=pretrained)

    # Change the first convolution layer
    # Original conv layer
    original_conv = model.conv1

    # New conv layer
    new_conv = torch.nn.Conv2d(input_chans, original_conv.out_channels, 
                            kernel_size=original_conv.kernel_size, 
                            stride=original_conv.stride, 
                            padding=original_conv.padding, 
                            bias=False)

    # Copy the weights from original conv layer to the new one
    if input_chans == 3:
        new_conv.weight.data = original_conv.weight.data
    elif input_chans < 3:
        new_conv.weight.data[:, 0:input_chans, :, :] = original_conv.weight.data[:, 0:input_chans, :, :]
    else:
        new_conv.weight.data[:, 0:3, :, :] = original_conv.weight.data

    # Replace the first conv layer
    model.conv1 = new_conv

    # If necessary, adjust the final layer based on the number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, output_size)
    return model


def get_unet(input_chans=7, output_size=1, pretrained=True):
    model = models.segmentation.fcn_resnet50(pretrained=pretrained)
    model.classifier[4] = torch.nn.Conv2d(512, output_size, kernel_size=(1, 1), stride=(1, 1))
    return model