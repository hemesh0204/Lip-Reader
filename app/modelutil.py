import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.initializers import Orthogonal



def load_model() -> tf.keras.Model:
    model_path = os.path.join('..', 'models', 'lipnet.keras')

    # Define Orthogonal explicitly in custom objects
    custom_objects = {'Orthogonal': Orthogonal}

    # Load the model using the custom objects dictionary
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    return model
