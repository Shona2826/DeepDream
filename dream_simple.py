import tensorflow as tf 
from deepdream import *
from Data_Preprocess import *
from pathlib import Path

def run_deep_dream_simple(img,steps = 100,step_size = 0.01):
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	img = tf.convert_to_tensor(img)
	step_size = tf.convert_to_tensor(step_size)
	steps_remaining = steps
	step = 0
	while steps_remaining:
		if steps_remaining > 100:
			run_steps = tf.constant(100)
		else:
			run_steps = tf.constant(steps_remaining)

		steps_remaining -= run_steps
		step += run_steps
		x = call_main()
		loss,img = x(img,run_steps,tf.constant(step_size))

		display.clear_output(wait = True)
		print("Step {}, loss {}".format(step, loss))

	result = deprocess(img)
	display.clear_output(wait= True)
	show(result)

	return result
