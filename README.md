# NAC-synthesis-trial

Draft process to synthesize a PyTorch model, using hls4ml & Vivado.

## Procedure

Install hls4ml.

Change `-maximum_size` to any value greater than 4608 from the 4096 that is currently in place, [here](https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/templates/vivado/build_prj.tcl#L164) (I doubled it to 8192).

Check the path of the file `model_weights_300_epochs.pth` with respect to the defined path in `run.py`.

Run `run.py` to load in the model, convert to hls, specify the config, and synthesize.

## Error

Still seems to freeze here:

`INFO: [HLS 200-489] Unrolling loop 'ResultLoop' (firmware/nnet_utils/nnet_conv2d_resource.h:96) in function 'nnet::conv_2d_cl<ap_fixed<16, 6, (ap_q_mode)5, (ap_o_mode)3, 0>, ap_fixed<16, 6, (ap_q_mode)5, (ap_o_mode)3, 0>, config2>' completely with a factor of 32`

## Note

I added this:
```
# Set io_type to 'io_stream' for the entire model
config['Model']['IOType'] = 'io_stream'

# Set strategy to 'Resource' for the entire model
config['Model']['Strategy'] = 'Resource'

# Set reuse factor to 32 for the entire model
config['Model']['ReuseFactor'] = 4 # changed from 32 to 4, crashed with 32
```
