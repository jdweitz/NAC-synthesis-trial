# Load in the pytorch model, convert to HLS, compile the HLS model, then synthesize the HLS model with Vivado

import torch
import torch.nn as nn
import numpy as np
import hls4ml

# Model architecture

class NAC(nn.Module):
    def __init__(self):
        super(NAC, self).__init__()
        # Define all layers in the model
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm2d = nn.BatchNorm2d(16)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*16, 32)
        self.gelu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(32, 32)
        self.gelu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(32, 16)
        self.batch_norm1d_16 = nn.BatchNorm1d(16)
        self.gelu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(16, 2)
        self.batch_norm1d_2 = nn.BatchNorm1d(2)

    def forward(self, x):
        # Connect the architecture components in forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.batch_norm2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.gelu2(x)
        x = self.fc3(x)
        x = self.batch_norm1d_16(x)
        x = self.gelu3(x)
        x = self.fc4(x)
        x = self.batch_norm1d_2(x)
        return x

# Define the model
model = NAC()

# Just use the randomly initialized weights; below is commented out

# # Model's state dictionary is saved in 'model_weights.pth'
# model_weights_path = 'removed_unquantized_LeakyReLU_NAC_iter5.pth'

# model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

model.eval()

# print(model)

# Generate initial configuration, altered here
config = hls4ml.utils.config_from_pytorch_model(model, granularity='model', inputs_channel_last=False, transpose_outputs=False)
# config = hls4ml.utils.config_from_pytorch_model(model, granularity='model')

# Set io_type to 'io_stream' for the entire model
config['Model']['IOType'] = 'io_stream'

# Set strategy to 'Resource' for the entire model
config['Model']['Strategy'] = 'Resource'

# Set reuse factor to 32 for the entire model
config['Model']['ReuseFactor'] = 4 # changed from 32 to 4, crashed with 8

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
    input_shape=(None, 1, 11, 11), # input_shape=(None, 1, 11, 11), # where 2048 if the batch size
    hls_config=config,
    output_dir='rf4_torch_model/hls4ml_prj',
    part='xcu250-figd2104-2L-e'
)

# Compile the HLS model
hls_model.compile()

# Synthesize (with Vivado)
hls_model.build(csim=False)

# Check the reports
hls4ml.report.read_vivado_report('rf4_torch_model/hls4ml_prj/')