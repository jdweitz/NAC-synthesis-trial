# NAC-synthesis-trial

Draft process to synthesize a PyTorch model, using hls4ml & Vivado.

## Procedure

Install hls4ml.

Change -maximum_size to any value greater than 4608 from the 4096 that is currently in place, [here](https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/templates/vivado/build_prj.tcl#L164) (I doubled it to 8192).

Check the path of the file of 'model_weights_300_epochs.pth'.

Run 'load_torch_model_convert_synthesize.py' to load in the model, convert to hls, specify parameters, and synthesize.

Open the Vivado report in the specified path.
