# coding=utf-8
# Copied from inference.py, to create a single encoder graph

import copy
import tensorflow as tf


def create_encoder_graph(models, features, params):
    if not isinstance(models, (list, tuple)):
        raise ValueError("'models' must be a list or tuple")
    if not len(models) == 1:
        raise ValueError("the length of 'models' must be 1")

    features = copy.copy(features)
    model_fns = [model.get_inference_func() for model in models]

    # Compute initial state if necessary
    # states = []
    # funcs = []

    model_fn = model_fns[0]
    # for model_fn in model_fns:
    if callable(model_fn):
        # For non-incremental decoding
        state = None
        # funcs.append(model_fn)
    else:
        # For incremental decoding where model_fn is a tuple:
        # (encoding_fn, decoding_fn)
        # state["encoder"] is the desired encoder output
        state = model_fn[0](features)
        # funcs.append(model_fn[1])

    if state is not None:
        encoder_outputs = state["encoder"]
    else:
        encoder_outputs = None

    return encoder_outputs
