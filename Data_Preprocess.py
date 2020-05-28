import tensorflow as tf 
import numpy as np 
import PIL.Image 
from tensorflow.keras.preprocessing import image 
import IPython.display as display
from pathlib import Path

def download(url,max_dim = None):
	name = 'pic1.png'
	image_path = url
	img = PIL.Image.open(image_path)
	if max_dim:
		img.thumbnail((max_dim,max_dim))
	return np.array(img)

#normalize an image
def deprocess(img):
	img = 255*(img+1.0)/2.0
	return tf.cast(img,tf.uint8)

#display the img
def show(img):
	shows = PIL.Image.fromarray(np.array(img))
	shows.show()



