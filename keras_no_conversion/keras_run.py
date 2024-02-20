import tensorflow as tf
from tensorflow.keras import layers, models
import hls4ml

# Define the sequential model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', input_shape=(11, 11, 1)),  # Initial conv layer

    # Block 1
    layers.Conv2D(4, 1, strides=1, padding='valid'),
    layers.ReLU(),
    layers.Conv2D(32, 1, strides=1, padding='valid'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    # Block 2
    layers.Conv2D(4, 1, strides=1, padding='valid'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Conv2D(32, 3, strides=1, padding='valid'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),

    # Block 3
    layers.Conv2D(8, 3, strides=1, padding='valid'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Conv2D(64, 3, strides=1, padding='valid'),

    layers.Flatten(),  # Flatten the output

    # MLP
    layers.Dense(8),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dense(4),
    layers.LeakyReLU(),
    layers.Dense(4),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dense(2)  # Final dense layer
])

model.build((None, 11, 11, 1))  # Input shape is (11, 11, 1)
model.summary()

model.compile()

# Generate initial configuration
config = hls4ml.utils.config_from_keras_model(model, granularity='model') # trying model that was made here (tf_model), instead of the loaded in one (loaded_tf_model)

# Now, manually adjust the generated configuration
# Since the granularity is set to 'model', these settings will apply model-wide

# Set io_type to 'io_stream' for the entire model
config['Model']['IOType'] = 'io_stream'

# Set strategy to 'Resource' for the entire model
config['Model']['Strategy'] = 'Resource'

# Set reuse factor to 32 for the entire model
config['Model']['ReuseFactor'] = 8 # changed from 32 to 8, crashed with 34

# Specify bit precision model-wide
config['Model']['Precision'] = 'ap_fixed<8,3>'

# Print the modified configuration
print("-----------------------------------")
print("Configuration")
print(config)  # Use print_dict(config) if you prefer a more structured output
print("-----------------------------------")

# Convert the PyTorch model to HLS model with the specified configuration
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    #input_shape=(1, 1, 11, 11), # PyTorch uses this, not tf
    hls_config=config,
    output_dir='synthesis_2/hls4ml_prj',
    part='xcu250-figd2104-2L-e'
)

# Compile the HLS model
hls_model.compile()

# Synthesize (with Vivado)
hls_model.build(csim=False)

# Check the reports
hls4ml.report.read_vivado_report('synthesis_2/hls4ml_prj/')
