"""wine dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(wine): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
The classic wine dataset.
* Augmented to have each feature first scaled between 0 and 1
* Augmented so that each instance appears d * 5 times where d is the number of features.
** This allows each instance to every combinations of scalings of its features.



        Alcohol

        Malic acid

        Ash

        Alcalinity of ash

        Magnesium

        Total phenols

        Flavanoids

        Nonflavanoid phenols

        Proanthocyanins

        Color intensity

        Hue

        OD280/OD315 of diluted wines

        Proline


"""

_CITATION = """

"""


class Wine(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for wine dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(wine): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'features': {
                'alcohol': tf.float32,
                'malic acid': tf.float32,
                'ash': tf.float32,
                'alcalinity of ash': tf.float32,
                'magnesium': tf.float32,
                'total phenols': tf.float32,
                'flavanoids': tf.float32,
                'nonflavanoid phenols': tf.float32,
                'proanthocyanins': tf.float64,
                'color intensity': tf.float32,
                'hue': tf.float32,
                'OD280/OD315 of diluted wines': tf.float32,
                'proline': tf.float32
            },
            'class': tf.int32
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('features', 'class'),  # e.g. ('image', 'label')
        homepage='https://archive.ics.uci.edu/ml/datasets/wine',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(wine): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')

    # TODO(wine): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(wine): Yields (key, example) tuples from the dataset
    yield 'key', {}
