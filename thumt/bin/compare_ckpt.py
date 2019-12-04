#!/usr/bin/env python
# coding=utf-8
# Modified (a lot) from translator.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import six
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.inference as inference
import thumt.utils.parallel as parallel
import thumt.utils.sampling as sampling


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the weights of two NMT models",
        usage="compare_ckpt.py [<args>] [-h | --help]"
    )

    # inputs
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")  # obviously we need 2 ckpt of the same model
    parser.add_argument("--metrics", type=str, nargs="+", default=["frobenius"],
                        help="The norms to calculate")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        device_list=[0],
        num_threads=1,
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    if model_name.startswith("experimental_"):
        model_name = model_name[13:]

    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix, feed_dict):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                placeholder = tf.placeholder(tf.float32,
                                             name="placeholder/" + var_name)
                with tf.device("/cpu:0"):
                    op = tf.assign(var, placeholder)
                    ops.append(op)
                feed_dict[placeholder] = value_dict[name]
                break

    return ops


def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]
        shard_size = (batch + num_shards - 1) // num_shards

        for i in range(num_shards):
            shard_feat = feat[i * shard_size:(i + 1) * shard_size]

            if shard_feat.shape[0] != 0:
                feed_dict[placeholders[i][name]] = shard_feat
                n = i + 1
            else:
                break

    if isinstance(predictions, (list, tuple)):
        predictions = predictions[:n]

    return predictions, feed_dict


def frobenius_distance(A, B):
    C = A - B
    if len(C.shape) == 1:
        C = np.dot(C, np.conjugate(C))
        return np.sqrt(np.sum(C))
    C = np.matmul(C, np.conjugate(C).T)
    return np.sqrt(np.trace(C))


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [models.get_model(args.model) for _ in range(len(args.checkpoints))]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoints[i], args.model, params_list[i])
        for i in range(len(args.checkpoints))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    model_var_lists = []

    # Load checkpoints
    for i, checkpoint in enumerate(args.checkpoints):
        tf.logging.info("Loading %s" % checkpoint)
        var_list = tf.train.list_variables(checkpoint)
        values = {}
        reader = tf.train.load_checkpoint(checkpoint)

        for (name, shape) in var_list:
            if not name.startswith(model_cls_list[i].get_name()):
                continue

            if name.find("losses_avg") >= 0:
                continue

            tensor = reader.get_tensor(name)
            values[name] = tensor
            tf.logging.info("Loading weight %s: %s" % (name, str(tensor.shape)))

        model_var_lists.append(values)

    distance = [dict()] * len(model_var_lists)
    # calculate distance
    for i in range(1, len(model_var_lists)):
        print("Comparing ckpt %s and %s" % (args.checkpoints[0], args.checkpoints[i]))
        for name in model_var_lists[i]:
            if "Adam" in name or "MultiStepOptimizer" in name:
                continue
            distance[i][name] = dict()
            for metric in args.metrics:
                if metric == "frobenius":
                    distance[i][name][metric] = frobenius_distance(
                        model_var_lists[0][name],
                        model_var_lists[i][name],
                    )
                else:
                    raise ValueError("Unknown metric %s" % metric)

    # Write to file
    if sys.version_info.major == 2:
        outfile = open(args.output, "w")
    elif sys.version_info.major == 3:
        outfile = open(args.output, "w", encoding="utf-8")
    else:
        raise ValueError("Unknown python running environment!")

    for i in range(1, len(model_var_lists)):
        outfile.write("Checkpoint %s VS %s\n" % (args.checkpoints[0], args.checkpoints[i]))
        for name in model_var_lists[i]:
            outfile.write("%s " % name)
            for metric in args.metrics:
                outfile.write("%f " % distance[i][name][metric])
            outfile.write("\n")

    outfile.close()


if __name__ == "__main__":
    main(parse_args())
