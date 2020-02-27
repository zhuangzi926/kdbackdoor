import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class Injector(keras.layers.Layer):
	"""inject backdoor into benign images

	Attributes:
		mask: a trainable tf Variable
		trigger: a tf constant
	"""

	def __init__(self, trigger, mask):
		"""generate injector layer
		
		Args:
			trigger: a numpy array of trigger if given, size = img_size, value in [0, 1)
			mask: a numpy array of mask if given, size = trig_size, value in [0, 1)
		"""

		super().__init__()
		self.mask = tf.Variable(
			initial_value=mask, 
			trainable=True, 
			dtype=tf.float32, 
			shape=mask.shape
		)
		self.trigger = tf.constant(
			value=trigger,
			dtype=tf.float32,
			shape=trigger.shape
		)
	
	def call(self, inputs):
		"""inject backdoor into an image

		backdoored_image = (1 - mask) * image + mask * trigger

		Args:
			inputs: a tensor of images, size = (batch_size, W, H, C), value in [0, 1]
		
		Returns:
			output: a tensor of backdoored images, size = (batch_size, W, H, C), value in [0, 1]
		"""

		output = (1 - self.mask) * inputs + self.mask * self.trigger
		output = tf.clip_by_value(output, 0., 1.)
		return output


class Backdoor(keras.Model):
	"""backdoor attack of neural network

	Attributes:
		trigger: a randomly initialized image
		mask: a randomly initialized mask ([0, 1])
		target_label: NN(mask(image, trigger)) = target_label, one-hot encoded
	"""

	def __init__(self, img_size=(28,28,1), trig_size=(28,28,1), 
				 class_num=10, target_label=None, 
				 trigger=None, mask=None):
		"""generate random trigger
		
		Args:
			img_size: a tuple indicates the size of image
			trig_size: a tuple indicates the size of trigger
			class_num: an int indicates total class numbers of dataset
			target_label: an int indicates the victim label
			trigger: a numpy array of trigger if given, size = img_size, value in [0, 1)
			mask: a numpy array of mask if given, size = trig_size, value in [0, 1)
		"""

		super().__init__()
		if target_label is None:
			self.target_label = np.random.randint(0, class_num)
		else:
			self.target_label = target_label
		self.target_label = tf.one_hot(self.target_label, class_num).numpy()

		if trigger is None:
			self.trigger = np.zeros(img_size)
			self.trigger[:trig_size[0], :trig_size[1], :trig_size[2]] = np.random.random(trig_size)
		else:
			self.trigger = trigger

		if mask is None:
			self.mask = np.random.random(trig_size)
		else:
			self.mask = mask
		
		self.layer = Injector(self.trigger, self.mask)

		# for debug
		print("target_label: ", self.target_label)
		print("trigger.shape: ", self.trigger.shape)
		print("mask.shape: ", self.mask.shape)
	
	def call(self, inputs):
		return self.layer(inputs)
		

if __name__ == "__main__":
	backdoor = Backdoor()
	print(backdoor.trainable_weights)

			
