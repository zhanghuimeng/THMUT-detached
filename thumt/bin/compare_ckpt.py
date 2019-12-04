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


def frobenius_distance(A, B):
    C = A - B
    if len(C.shape) == 1:
        C = np.dot(C, np.conjugate(C))
        return np.sqrt(np.sum(C))
    C = np.matmul(C, np.conjugate(C).T)
    return np.sqrt(np.trace(C))


def euclidean_distance(A, B):
    C = A - B
    C = C * C
    return np.sqrt(np.sum(C))


def d1_norm(A, B):
    C = A - B
    C = np.abs(C)
    return np.sum(C)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [models.get_model(args.model) for _ in range(len(args.checkpoints))]

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

    dist = dict()
    # calculate distance
    for i in range(1, len(model_var_lists)):
        print("Comparing ckpt %s and %s" % (args.checkpoints[0], args.checkpoints[i]))
        for name in model_var_lists[i]:
            if "Adam" in name or "MultiStepOptimizer" in name or "encoder" in name:
                continue
            for metric in args.metrics:
                if model_var_lists[0][name].shape != model_var_lists[i][name].shape:
                    continue
                if metric == "frobenius":
                    dist[(metric, i, name)] = frobenius_distance(
                        model_var_lists[0][name],
                        model_var_lists[i][name],
                    )
                elif metric == "euclidean":
                    dist[(metric, i, name)] = euclidean_distance(
                        model_var_lists[0][name],
                        model_var_lists[i][name],
                    )
                elif metric == "d1_norm":
                    dist[(metric, i, name)] = d1_norm(
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

    outfile.write(",")
    for i in range(1, len(model_var_lists)):
        outfile.write("%s," % args.checkpoints[i])
    outfile.write("\n")
    for name in model_var_lists[0]:
        if "Adam" in name or "MultiStepOptimizer" in name or "encoder" in name:
            continue
        if "source_embedding" in name or "transformer/bias" in name:
            continue
        outfile.write("%s," % name)
        for i in range(1, len(model_var_lists)):
            for metric in args.metrics:
                if (metric, i, name) in dist:
                    outfile.write("%f," % dist[(metric, i, name)])
                else:
                    outfile.write("-1,")
        outfile.write("\n")

    outfile.close()


if __name__ == "__main__":
    main(parse_args())
