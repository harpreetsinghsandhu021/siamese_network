
import tensorflow as tf
from tensorflow.keras.layers import Layer


class L1Dist(Layer):
    # Init Method - Inheritance
    def __init__(self, **kwargs):
        super().__init__()
    
    # Similarity Calculation
    def call(self, input_embedding, validation_embedding): 
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)