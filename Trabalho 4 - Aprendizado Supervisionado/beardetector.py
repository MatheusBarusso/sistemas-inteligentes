# Daniel Cavalcanti Jeronymo
# 09/2020
# MobileNet example for detecting bears

#from tensorflow.keras.applications.mobilenetv2 import MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet as MobileNetV2 # ugly, ugly hack
from tensorflow.keras.layers import Input
import PIL
import numpy as np
import urllib
import os

# Code adapted from:
# https://github.com/JonathanCMitchell/mobilenet_v2_keras/blob/master/imagenet_utils.py
def create_readable_names_for_imagenet_labels():
  """Create a dict mapping label id to human readable string.
  Returns:
      labels_to_names: dictionary where keys are integers from to 1000
      and values are human-readable names.
  We retrieve a synset file, which contains a list of valid synset labels used
  by ILSVRC competition. There is one synset one per line, eg.
          #   n01440764
          #   n01443537
  We also retrieve a synset_to_human_file, which contains a mapping from synsets
  to human-readable names for every synset in Imagenet. These are stored in a
  tsv format, as follows:
          #   n02119247    black fox
          #   n02119359    silver fox
  We assign each synset (in alphabetical order) an integer, starting from 1
  (since 0 is reserved for the background class).
  Code is based on
  https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py#L463
  """

  # pylint: disable=g-line-too-long
  synset_url = 'https://raw.githubusercontent.com/tensorflow/models/1af55e018eebce03fb61bba9959a04672536107d/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt'
  synset_to_human_url = 'https://raw.githubusercontent.com/tensorflow/models/1af55e018eebce03fb61bba9959a04672536107d/research/slim/datasets/imagenet_metadata.txt'

  filename, _ = urllib.request.urlretrieve(synset_url)
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  assert num_synsets_in_ilsvrc == 1000

  filename, _ = urllib.request.urlretrieve(synset_to_human_url)
  synset_to_human_list = open(filename).readlines()
  num_synsets_in_all_imagenet = len(synset_to_human_list)
  assert num_synsets_in_all_imagenet == 21842

  synset_to_human = {}
  for s in synset_to_human_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human

  label_index = 1
  labels_to_names = {0: 'background'}
  for synset in synset_list:
    name = synset_to_human[synset]
    labels_to_names[label_index] = name
    label_index += 1

  return labels_to_names

def main(filename, rows=600, alpha=1.0):
    # Preprocess image
    img = np.array(PIL.Image.open(filename).resize((rows, rows))).astype(np.float32) / 128. - 1.
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Create input layer
    inputTensor = Input(shape=(rows, rows, 3))

    # Note: you could also provide an input_shape
    model = MobileNetV2(input_tensor=inputTensor, include_top=True, weights='imagenet', alpha=alpha)

    # Get output activations
    y = model.predict(img)
    y = y[0].ravel()

    # Convert output to text
    label_map = create_readable_names_for_imagenet_labels()
    #yIndex = y.argmax() + 1 # 0 is background, shift over by 1
    
    # Order multiple best outputs
    # first get best indices then sort them and invert
    nBest = 4
    yIndex = np.argpartition(y, -nBest)[-nBest:]
    yIndex = yIndex[np.argsort(y[yIndex])][::-1] + 1 # offset by 1

    print('\n=============')
    print('Input image {} is: '.format(filename))
    print(label_map[yIndex[0]])

    print('\nIt could also be:')
    print(*[label_map[i] for i in yIndex[1:]], sep='\n')


if __name__ == "__main__":
    # Set the working directory (where we expect to find files) to the same
    # directory this .py file is in. You can leave this out of your own
    # code, but it is needed to easily run the examples using "python -m"
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    main('urso_marrom_wikipedia.jpg')
    main('ursinho_carinhoso1.jpg')
    main('ursinho_carinhoso2.jpg')