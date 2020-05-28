import tensorflow as tf 
def calc_loss(img,model):
	img_batch = tf.expand_dims(img,axis = 0)
	layer_activations = model(img_batch)
	if len(layer_activations) == 1:
		layer_activations = [layer_activations]

	losses = []
	for act in layer_activations:
		loss = tf.math.reduce_mean(act)
		losses.append(loss)
	return tf.reduce_sum(losses)