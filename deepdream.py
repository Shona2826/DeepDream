import tensorflow as tf 
from calculate_loss import *
from feature_extraction import dream,feature_extractions
class DeepDream(tf.Module):
	def __init__(self,model):
		self.model = model

	@tf.function(
		input_signature = (
			tf.TensorSpec(shape = [None,None,3],dtype = tf.float32),
			tf.TensorSpec(shape = [],dtype = tf.int32),
			tf.TensorSpec(shape	=[],dtype = tf.float32),))

	def __call__(self,img,steps,step_size):
		print("Tracing")
		loss = tf.constant(0.0)
		for n in tf.range(steps):
			with tf.GradientTape() as tape:
				tape.watch(img)
				loss = calc_loss(img,self.model)
			gradients = tape.gradient(loss,img)

			gradients /= tf.math.reduce_std(gradients) + 1e-8

			img = img + gradients * step_size
			img = tf.clip_by_value(img,-1,1)
		return loss,img

def call_main():
	x,y = feature_extractions()
	z = dream(x,y)
	deepdream = DeepDream(z)
	return deepdream