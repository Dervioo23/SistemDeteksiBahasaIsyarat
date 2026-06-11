import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf


def set_random_seeds(seed: int = 42) -> None:
    """Atur seed acak untuk pelatihan yang dapat direproduksi.

    Ini mempengaruhi modul random Python, NumPy, TensorFlow, dan PYTHONHASHSEED.
    Catatan: Determinisme penuh juga bergantung pada pengaturan GPU/cudnn, tetapi ini
    secara signifikan mengurangi varians antar-jalan.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
