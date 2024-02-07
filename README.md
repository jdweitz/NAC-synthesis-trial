# NAC-synthesis-trial

Draft process to synthesize a PyTorch model, using hls4ml & Vivado.

## Procedure

Install hls4ml.

Change `-maximum_size` to any value greater than 4608 from the 4096 that is currently in place, [here](https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/templates/vivado/build_prj.tcl#L164), at your respective file (I doubled it to 8192).

Check the path of the file `model_weights_300_epochs.pth` with respect to the defined path in `run.py`.

Run `run.py` to load in the model, convert to hls, specify the config, and synthesize.

## Note 1

I added this to `run.py`:
```
# Set io_type to 'io_stream' for the entire model
config['Model']['IOType'] = 'io_stream'

# Set strategy to 'Resource' for the entire model
config['Model']['Strategy'] = 'Resource'

# Set reuse factor to 32 for the entire model
config['Model']['ReuseFactor'] = 4 # changed from 32 to 4, crashed with 32
```

The crash occurred when evaluating the the hls model (separate from files in this repo, as it requires loading in the data), so I changed to 4 and it evaluated with only this note:
`WARNING: Invalid ReuseFactor=4 in layer "conv1".Using ReuseFactor=3 instead. Valid ReuseFactor(s): 1,3,9,18,36,72,144,288.` which is no issue. 

Also, got a much smaller discrepancy between the PyTorch and hls model with these parameters:
```
HLS Mean Distance:  0.39590603
Mean Distance:  0.40206024
```

## Note 2

Unrolling here takes ~ 20min (this was the previous bottleneck layer 4608 > 4096)

`INFO: [HLS 200-489] Unrolling loop 'ResultLoop' (firmware/nnet_utils/nnet_conv2d_resource.h:96) in function 'nnet::conv_2d_cl<ap_fixed<16, 6, (ap_q_mode)5, (ap_o_mode)3, 0>, ap_fixed<16, 6, (ap_q_mode)5, (ap_o_mode)3, 0>, config2>' completely with a factor of 32`

## Note 3

Had this happen with 30 different arrays:

`
INFO: [XFORM 203-101] Partitioning array 'layer17_out.V' (firmware/myproject.cpp:101) in dimension 1 completely.
`

`
WARNING: [XFORM 203-104] Completely partitioning array 'layer17_out.V' (firmware/myproject.cpp:101) accessed through non-constant indices on dimension 1 (firmware/nnet_utils/nnet_array.h:43:17), which may result in long runtime and suboptimal QoR due to large multiplexers. Please consider wrapping the array access into a function or using a register file core instead.
`
