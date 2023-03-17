"""
universal sentence encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader
import tensorflow as tf


hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")


class UniversalSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        else:
            tfhub_url = "/data/webw6/glensjh/datas/universal-sentence-encoder_4"
            #tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

        self._tfhub_url = tfhub_url
        # Lazily load the model
        self.model = None

    def encode(self, sentences):
        with tf.device('/CPU:0'):
            if not self.model:
                self.model = hub.load(self._tfhub_url)

            return tf.stop_gradient(self.model(sentences)).numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None
