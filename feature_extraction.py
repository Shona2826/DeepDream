import tensorflow as tf 
from tensorflow.keras.preprocessing import image

def feature_extractions():
	base_model = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet')
	names = ['mixed3','mixed3']
	layers = [base_model.get_layer(name).output for name in names]
	return base_model,layers
	#extraction model

def dream(base_model,layers):
	dream_model = tf.keras.Model(inputs=base_model.input,outputs = layers)
	return dream_model
