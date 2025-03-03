import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLPd1T1(nn.Module):
    def __init__(self, input_dim, out_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.dense1 = nn.Linear(in_features=self.input_dim , out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=64)
        self.dense3 = nn.Linear(in_features=64, out_features=self.out_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.tanh(self.dense1(x))
        x = self.tanh(self.dense2(x))
        # x = self.sigmoid(self.dense3(x))
        x = self.dense3(x)
        return x    

class Generator(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, batch_size, sequence_length, out_dim, device) -> None:
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.out_dim = out_dim
        self.device = device

        self.lstm1 = nn.LSTM(self.input_features, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.reshape = Reshape(self.sequence_length, self.hidden_size)
        self.dense1 = nn.Linear(in_features=2*self.sequence_length*self.hidden_size, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=self.out_dim)
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, use_interpolation, interpolation_steps=None):
        current_batch_size = x.shape[0]
        feature_dim = x.shape[2]
        x,_ = self.lstm1(x)
        x = self.reshape(x)
        x = self.activation(self.dense1(x))
        x = self.dense2(x)
        if use_interpolation:
            x = x.reshape(current_batch_size, feature_dim, 1)
            x = nn.functional.interpolate(x, size=interpolation_steps, mode="linear")
            x = torch.transpose(x, dim0=2, dim1=1).reshape(-1, feature_dim)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim ,output_dim) -> None:
        super().__init__()
        self.channel_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(in_channels=self.channel_dim, out_channels=32, kernel_size=2, padding=1, stride=2)
        self.activation = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm1d(num_features=32, affine=True)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, padding=1, stride=2)
        self.bn2 = nn.InstanceNorm1d(num_features=64, affine=True)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, padding=1, stride=2)
        self.bn3 = nn.InstanceNorm1d(num_features=128, affine=True)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, padding=1, stride=2)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=2, padding=0, stride=2)

    def forward(self,x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class Reshape(nn.Module):
    def __init__(self, sequence_length, hidden_size):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

    def forward(self, x):
        return x.reshape(x.shape[0], -1)
    
class rnn_forwardModelD1(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, batch_size, sequence_length, out_dim, device) -> None:
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.out_dim = out_dim
        self.device = device

        self.lstm1 = nn.LSTM(self.input_features, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.reshape = Reshape(self.sequence_length, self.hidden_size)
        self.dense1 = nn.Linear(in_features=2*self.sequence_length*self.hidden_size, out_features=50)
        self.dense2 = nn.Linear(in_features=50, out_features=self.out_dim)
        self.activation = nn.ReLU()
        
        self.tanh = nn.Tanh()

    def forward(self,x):
        x,_ = self.lstm1(x)
        x = self.reshape(x)
        x = self.tanh(self.dense1(x))
        x = self.dense2(x)
        return x
    
class Reshape(nn.Module):
    def __init__(self, sequence_length, hidden_size):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

    def forward(self, x):
        return x.reshape(x.shape[0], -1)