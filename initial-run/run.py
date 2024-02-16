# Load in the pytorch model, convert to HLS, compile the HLS model, then synthesize the HLS model with Vivado

import torch
import torch.nn as nn
import numpy as np
import hls4ml

# Model architecture

class NAC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 4, 1, 1), #nn.Linear(32, 4), #Point-wise Convolution
            nn.ReLU(),
            nn.Conv2d(4, 32, 1, 1), #nn.Linear(4, 32), #Point-wise Convolution
            nn.BatchNorm2d(32),
            nn.LeakyReLU() #nn.LeakyReLU() (tried out to see if would get a smaller delta is used relu - not the case)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 4, 1, 1), #nn.Linear(32, 4), #Point-wise Convolution
            nn.BatchNorm2d(4),
            nn.LeakyReLU(), #nn.GELU(), (hls4ml not supported)
            nn.Conv2d(4, 32, 3, 1),
            nn.BatchNorm2d(32), #nn.LayerNorm((32, 7, 7)), (hls4ml not supported)
            nn.LeakyReLU() #nn.GELU() (hls4ml not supported)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1),
            nn.BatchNorm2d(8), #nn.LayerNorm((8, 5, 5)), (hls4ml not supported)
            nn.LeakyReLU(), #nn.GELU(), (hls4ml not supported)
            nn.Conv2d(8, 64, 3, 1),
        )
        self.flatten = nn.Flatten() # added this for hls4ml support
        self.mlp = nn.Sequential(
            nn.Linear(576, 8),
            nn.BatchNorm1d(8), #nn.LayerNorm((8)), (hls4ml not supported)
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(), #nn.GELU(), (hls4ml not supported)
            nn.Linear(4,4),
            nn.BatchNorm1d(4), #nn.LayerNorm((4)), (hls4ml not supported)
            nn.LeakyReLU(), #nn.GELU(), (hls4ml not supported)
            nn.Linear(4,2),
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x) #torch.flatten(x,1) # removed this as hls4ml does not support it
        x = self.mlp(x)
        return x

# Define the model
model = NAC()

# Model's state dictionary is saved in 'model_weights.pth'
model_weights_path = 'model_weights_300_epochs.pth'

model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

model.eval()

# Generate initial configuration
config = hls4ml.utils.config_from_pytorch_model(model, granularity='model')  # name is not working properly

# Set io_type to 'io_stream' for the entire model
config['Model']['IOType'] = 'io_stream'

# Set strategy to 'Resource' for the entire model
config['Model']['Strategy'] = 'Resource'

# Set reuse factor to 32 for the entire model
config['Model']['ReuseFactor'] = 8 # changed from 32 to 8, crashed with 32

# Print the modified configuration
print("-----------------------------------")
print("Configuration")
print(config)  # Use print_dict(config) if you prefer a more structured output
print("-----------------------------------")

# Convert the PyTorch model to HLS model with the specified configuration
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    input_shape=(None, 1, 11, 11),
    hls_config=config,
    output_dir='model_draft/hls4ml_prj',
    part='xcu250-figd2104-2L-e'
)

# Compile the HLS model
hls_model.compile()

# Synthesize (with Vivado)
hls_model.build(csim=False)

# Check the reports
hls4ml.report.read_vivado_report('model_draft/hls4ml_prj/')
