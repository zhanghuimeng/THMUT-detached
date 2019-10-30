#!/usr/bin/env python
# coding=utf-8
# Load model weights and a series of (Deutsch) sentences, and output the hidden representation produced by
# THUMT encoder.

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
import thumt.utils.encoder as encoder
import thumt.utils.parallel as parallel
import thumt.utils.sampling as sampling


def parse_args():
    parser = argparse.ArgumentParser(
        description="Output hidden representation of source side using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input (monolingual) file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    # Only one checkpoint now
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path of trained model")
    parser.add_argument("--vocabulary", type=str, nargs=1, required=True,
                        help="Path of source vocabulary")

    # model and configuration
    # Only one model now
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        device_list=[0],
        num_threads=1,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        decode_batch_size=32,  # It means source batch size...
        # sampling
        generate_samples=False,
        num_samples=1,
        min_length_ratio=0.0,
        max_length_ratio=1.5,
        min_sample_length=0,
        max_sample_length=0,
        sample_batch_size=32
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
    # Deleted target vocabulary
    if args.parameters:
        params.parse(args.parameters)

    # Actually vocab is a simple list
    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        # "target": vocabulary.load_vocabulary(args.vocabulary[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    # params.vocabulary["target"] = vocabulary.process_vocabulary(
    #     params.vocabulary["target"], params
    # )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    # Although I don't know what is mapping useful for
    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        # "target": vocabulary.get_control_mapping(
        #     params.vocabulary["target"],
        #     control_symbols
        # )
    }

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


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    # Only load one config
    model_cls_list = [models.get_model(args.model)]
    params_list = [default_parameters()]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoint, args.model, params_list[0])
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        # Only one checkpoint now
        # for i, checkpoint in enumerate(args.checkpoints):
        tf.logging.info("Loading %s" % args.checkpoint)
        var_list = tf.train.list_variables(args.checkpoint)
        values = {}
        reader = tf.train.load_checkpoint(args.checkpoint)

        for (name, shape) in var_list:
            if not name.startswith(model_cls_list[0].get_name()):
                continue

            if name.find("losses_avg") >= 0:
                continue

            tensor = reader.get_tensor(name)
            values[name] = tensor

        # values is name:weight value pairs
        model_var_lists.append(values)

        # Build models
        model_list = []

        # for i in range(len(args.checkpoints)):
        name = model_cls_list[0].get_name()
        model = model_cls_list[0](params_list[0], name + "_%d" % 0)
        model_list.append(model)

        # debug, print checkpoint
        # print(model_list[0])
        # for key in model_var_lists[0]:
        #     print(key, model_var_lists[0][key].shape)

        params = params_list[0]
        # Read input file
        sorted_keys, sorted_inputs = dataset.sort_input_file(args.input)
        # Build input queue
        features = dataset.get_inference_input(sorted_inputs, params)
        # Create placeholders
        placeholders = []

        for i in range(len(params.device_list)):
            placeholders.append({
                "source": tf.placeholder(tf.int32, [None, None],
                                         "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None],
                                                "source_length_%d" % i)
            })

        # A list of outputs
        # if params.generate_samples:
        #     inference_fn = sampling.create_sampling_graph
        # else:
        #     inference_fn = inference.create_inference_graph
        # predictions = parallel.data_parallelism(
        #     params.device_list, lambda f: inference_fn(model_list, f, params),
        #     placeholders)

        # I don't need predictions, only encoder output results
        encoder_fn = encoder.create_encoder_graph
        encoder_output = parallel.data_parallelism(
            params.device_list, lambda f: encoder_fn(model_list, f, params),
            placeholders)

        # Create assign ops
        assign_ops = []
        feed_dict = {}

        all_var_list = tf.trainable_variables()
        # debug, print all var list
        # for var in all_var_list:
        #     print(var)

        # for i in range(len(args.checkpoints)):
        un_init_var_list = []
        name = model_cls_list[0].get_name()

        for v in all_var_list:
            if v.name.startswith(name + "_%d" % 0):
                un_init_var_list.append(v)

        ops = set_variables(un_init_var_list, model_var_lists[0],
                            name + "_%d" % 0, feed_dict)
        assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)
        init_op = tf.tables_initializer()
        results = []

        tf.get_default_graph().finalize()

        # Create session
        with tf.Session(config=session_config(params)) as sess:
            # Restore variables
            sess.run(assign_op, feed_dict=feed_dict)
            sess.run(init_op)

            while True:
                try:
                    feats = sess.run(features)
                    # op here is actually predictions
                    op, feed_dict = shard_features(feats, placeholders,
                                                   encoder_output)
                    results.append(sess.run(op, feed_dict=feed_dict))
                    message = "Finished batch %d" % len(results)
                    tf.logging.log(tf.logging.INFO, message)
                except tf.errors.OutOfRangeError:
                    break

        # print encoder_outputs
        encoder_outputs = []

        # TODO: the inputs are sorted.

        # each shard saved the values returned by create_encoder_graph (create_inference_graph)
        for result in results:
            # the shard does not have extra length when there is not more than one return values
            for shard in result:
                print(shard.shape)
                # for item in shard:
                # print(item.shape)
                encoder_outputs.append(shard)
        # stack the batches up
        # encoder_outputs = np.concatenate(encoder_outputs, axis=0)
        # print(encoder_outputs.shape)

        # Write to file
        # np.savetxt(args.output, encoder_outputs, delimiter=",")


if __name__ == "__main__":
    main(parse_args())
