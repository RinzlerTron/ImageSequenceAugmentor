from absl import app
from absl import flags
from absl import logging

import keras
import os
import numpy as np
import cv2
from tempfile import NamedTemporaryFile

logging.set_verbosity(logging.INFO)

flags.DEFINE_string('data_url', '', 'URL with image data in npz file')
flags.DEFINE_string('export_path', None, 'path to save output images')
flags.DEFINE_string('label_sequence', '', 'label sequence to be generated')
flags.DEFINE_integer('image_width', 32, 'width of image in pixels')
flags.DEFINE_integer('min_spacing', 10,
                     'minimum spacing between images in pixel')
flags.DEFINE_integer('max_spacing', 100,
                     'maximum spacing between images in pixel')
FLAGS = flags.FLAGS


def load_image_data():
  """Load image data from npz file"""
  path = keras.utils.get_file(os.path.basename(FLAGS.data_url), FLAGS.data_url)
  # Combine training and testing data
  with np.load(path) as data:
    example_values = np.concatenate((data['x_train'], data['x_test']),
                                    axis = 0)
    example_labels = np.concatenate((data['y_train'], data['y_test']),
                                    axis = 0)
  return example_values, example_labels

def get_label_images(label_sequence_input, example_values, example_labels):
  """Get image data for each label in input label sequence."""
  sequence_values = [int(x) for x in label_sequence_input]
  randomized_indices = []
  for i in sequence_values:
    # Get random index where label value matches required image
    randomized_indices.append(np.random.choice(
        np.argwhere(example_labels==i).flatten(), replace=False))
  return example_values[randomized_indices]

def resize_image(image_array, image_width):
    """Resize image width and height based on image width input"""
    return cv2.resize(image_array, (image_width,image_width),
                      interpolation=cv2.INTER_CUBIC)

def create_label_sequence(label, image_width, min_spacing, max_spacing):
  """ A function to create an image representing the given labels,
  with random spacing between the images.
  Each image is randomly sampled from the supplied dataset.
  Returns an NumPy array representing the image.
  Args:
    label: A string representing the sequence of images, e.g. "14543"
    image_width: The image width (in pixel).
    min_spacing: The minimum spacing between images (in pixel).
    max_spacing: The maximum spacing between images (in pixel).
  Returns:
    image array representing sequence of input labels
  """
  sequence_image_arrays = get_label_images(label, *load_image_data())

  def add_random_spacing(image_array):
    """Pad input image array with random spacing on right side."""
    return np.pad(image_array, ((0, 0), # top, bottom, left, right
                (0, np.random.randint(min_spacing, max_spacing))),
                mode='constant', constant_values=0)
  # Stack image arrays for images to form array for label sequence
  combined_image_arrays = np.hstack((np.asarray
                                     (add_random_spacing(image))
                                     for image in sequence_image_arrays))
  # Resize image to match width requirement
  if image_width:
    combined_image_arrays = resize_image(combined_image_arrays, image_width)
  return combined_image_arrays


def main(unused_argv):
  image_array = create_label_sequence(FLAGS.label, FLAGS.image_width,
                        FLAGS.min_spacing, FLAGS.max_spacing)
  output_file = NamedTemporaryFile(prefix='outfile', dir=FLAGS.export_path)
  # save output file with image array and sequence label
  np.savez(output_file, sequence_values=image_array,
           sequence_label=FLAGS.label)
  print('output saved in {}'.format(output_file))

if __name__ == '__main__':
  app.run(main)
