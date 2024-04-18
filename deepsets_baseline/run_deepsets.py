# Load in the pytorch model, convert to HLS, compile the HLS model, then synthesize the HLS model with Vivado

import torch
import torch.nn as nn
import numpy as np
import hls4ml

# Model architecture

class DeepSetsInv(nn.Module):
    def __init__(self, input_size, nnodes_phi: int = 32, nnodes_rho: int = 16, activ: str = "relu"):
        super(DeepSetsInv, self).__init__()
        self.nclasses = 5
        
        self.phi = nn.Sequential(
            nn.Linear(input_size, nnodes_phi),
            self.get_activation(activ),
            nn.Linear(nnodes_phi, nnodes_phi),
            self.get_activation(activ),
            nn.Linear(nnodes_phi, nnodes_phi),
            self.get_activation(activ),
        )
        
        self.rho = nn.Sequential(
            nn.Linear(nnodes_phi, nnodes_rho),
            self.get_activation(activ),
            nn.Linear(nnodes_rho, self.nclasses),
        )
    
    def get_activation(self, activ):
        if activ == "relu":
            return nn.ReLU()
        elif activ == "sigmoid":
            return nn.Sigmoid()
        elif activ == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activ}")
    
    def forward(self, inputs):
        phi_output = self.phi(inputs)
        # print("phi_output dtype:", phi_output.dtype)
        sum_output = torch.mean(phi_output, dim=1)
        # sum_output = torch.sum(phi_output, dim=1) / phi_output.shape[1]
        # # Manual computation of mean without using torch.mean or torch.sum
        # sum_output = torch.zeros(phi_output.size(0), phi_output.size(2), device=phi_output.device, dtype=phi_output.dtype)
        # for i in range(phi_output.size(0)):  # Loop over the batch
        #     for j in range(phi_output.size(2)):  # Loop over the dimension that we're averaging across
        #         sum_temp = 0.0
        #         for k in range(phi_output.size(1)):  # Loop over the dimension to sum
        #             sum_temp += phi_output[i, k, j]
        #         sum_output[i, j] = sum_temp / phi_output.size(1)
        # print( "sum_output dtype:", sum_output.dtype)
        rho_output = self.rho(sum_output)
        # print( "rho_output dtype:", rho_output.dtype)
        return rho_output

# Define the model
input_size = 3  # Assuming each input feature vector has a size of 16
model = DeepSetsInv(input_size=input_size, nnodes_phi=32, nnodes_rho=16, activ="relu")

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

# # Specify bit precision model-wide
# config['Model']['Precision'] = 'ap_fixed<8,3>'

# Print the modified configuration
print("-----------------------------------")
print("Configuration")
print(config)  # Use print_dict(config) if you prefer a more structured output
print("-----------------------------------")

# Convert the PyTorch model to HLS model with the specified configuration
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    input_shape=(32, 8, 3), # (batch, # constituents, # features)
    hls_config=config,
    output_dir='rf4_torch_model/hls4ml_prj',
    part='xcu250-figd2104-2L-e'
)

# Compile the HLS model
hls_model.compile()

# # Synthesize (with Vivado)
# hls_model.build(csim=False)

# # Check the reports
# hls4ml.report.read_vivado_report('rf4_torch_model/hls4ml_prj/')