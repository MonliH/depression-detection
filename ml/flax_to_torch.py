#!/usr/bin/env python
import tempfile
import sys

import jax
from jax import numpy as jnp
from transformers import AutoTokenizer, FlaxBigBirdForSequenceClassification, BigBirdForSequenceClassification


def to_f32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)



def main(model_path, new_path):
    # Saving extra files from config.json and tokenizer.json files
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(new_path)

    # Temporary saving bfloat16 Flax model into float32
    tmp = tempfile.mkdtemp()
    flax_model = FlaxBigBirdForSequenceClassification.from_pretrained(model_path)
    flax_model.params = to_f32(flax_model.params)
    flax_model.save_pretrained(tmp)
    # Converting float32 Flax to PyTorch
    model = BigBirdForSequenceClassification.from_pretrained(tmp, from_flax=True)
    model.save_pretrained(new_path, save_config=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
