# Load in the pytorch model, convert to HLS, compile the HLS model, then synthesize the HLS model with Vivado

# test change
# another test change

import torch
import torch.nn as nn
import numpy as np
import hls4ml

# Model architecture

class NAC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.act1 = nn.ReLU()
        self.conv3 = nn.Conv2d(4, 32, kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.act2 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(4)
        self.act3 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(4, 32, kernel_size=3, stride=1)
        self.norm3 = nn.BatchNorm2d(32) #nn.LayerNorm((32, 7, 7))
        self.act4 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(32, 8, kernel_size=3, stride=1)
        self.norm4 = nn.BatchNorm2d(8) #nn.LayerNorm((8, 5, 5))
        self.act5 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(8, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(576, 8)
        self.norm5 = nn.BatchNorm1d(8) #nn.LayerNorm((8))
        self.act6 = nn.ReLU()
        self.fc2 = nn.Linear(8, 4)
        self.act7 = nn.LeakyReLU()
        self.fc3 = nn.Linear(4,4)
        self.norm6 = nn.BatchNorm1d(4) #nn.LayerNorm((4))
        self.act8 = nn.LeakyReLU()
        self.fc4 = nn.Linear(4,2)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.norm1(x)
        x = self.act2(x)
        x = self.conv4(x)
        x = self.norm2(x)
        x = self.act3(x)
        x = self.conv5(x)
        x = self.norm3(x)
        x = self.act4(x)
        x = self.conv6(x)
        x = self.norm4(x)
        x = self.act5(x)
        x = self.conv7(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.norm5(x)
        x = self.act6(x)
        x = self.fc2(x)
        x = self.act7(x)
        x = self.fc3(x)
        x = self.norm6(x)
        x = self.act8(x)
        x = self.fc4(x)
        return x

# Define the model
model = NAC()

# Model's state dictionary is saved in 'model_weights.pth'
model_weights_path = 'removed_unquantized_LeakyReLU_NAC_iter5.pth'

model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

model.eval()

# Generate initial configuration, altered here
config = hls4ml.utils.config_from_pytorch_model(model, granularity='model', inputs_channel_last=False, transpose_outputs=False)
# config = hls4ml.utils.config_from_pytorch_model(model, granularity='model')

# Set io_type to 'io_stream' for the entire model
config['Model']['IOType'] = 'io_stream'

# Set strategy to 'Resource' for the entire model
config['Model']['Strategy'] = 'Resource'

# Set reuse factor to 32 for the entire model
config['Model']['ReuseFactor'] = 8 # changed from 32 to 4, crashed with 8

# Specify bit precision model-wide
config['Model']['Precision'] = 'ap_fixed<8,3>'

# Print the modified configuration
print("-----------------------------------")
print("Configuration")
print(config)  # Use print_dict(config) if you prefer a more structured output
print("-----------------------------------")

# Convert the PyTorch model to HLS model with the specified configuration
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    input_shape=(None, 11, 11, 1), # input_shape=(None, 1, 11, 11), # where 2048 if the batch size
    hls_config=config,
    output_dir='torch_model/hls4ml_prj',
    part='xcu250-figd2104-2L-e'
)

# Compile the HLS model
hls_model.compile()

# Synthesize (with Vivado)
hls_model.build(csim=False)

# Check the reports
hls4ml.report.read_vivado_report('model/hls4ml_prj/')