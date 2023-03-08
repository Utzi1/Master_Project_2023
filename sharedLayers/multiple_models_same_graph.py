import tensorflow.keras as ks
from tensorflow.keras import layers


encoder_input = ks.Input(shape=(82, 28, 1), name="img")
# and build a generic stack of layers:
layer_stack = layers.Conv2D(16, 3, activation="relu")(encoder_input)
layer_stack = layers.Conv2D(32, 3, activation="relu")(encoder_input)
layer_stack = layers.MaxPool2D(3)(layer_stack)
layer_stack = layers.Conv2D(32, 3, activation="relu")(layer_stack)
layer_stack = layers.Conv2D(16, 3, activation="relu")(layer_stack)
encoder_output = layers.GlobalMaxPooling2D()(layer_stack)

encoder = ks.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

layer_stack = layers.Reshape((4, 4, 1))(encoder_output)
layer_stack = layers.Conv2DTranspose(16, 3, activation="relu")(layer_stack)
layer_stack = layers.Conv2DTranspose(16, 3, activation="relu")(layer_stack)
layer_stack = layers.UpSampling2D(3)(layer_stack)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(layer_stack)

autencoder = ks.Model(encoder_input, decoder_output, name="autencoder")
autencoder.summary()
