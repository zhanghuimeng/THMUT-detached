# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import operator
import os

import tensorflow as tf
import thumt.utils.bleu as bleu


def _get_saver():
    # Get saver from the SAVERS collection if present.
    collection_key = tf.GraphKeys.SAVERS
    savers = tf.get_collection(collection_key)

    if not savers:
        raise RuntimeError("No items in collection {}. "
                           "Please add a saver to the collection ")
    elif len(savers) > 1:
        raise RuntimeError("More than one item in collection")

    return savers[0]


def _save_log(filename, result):
    metric, global_step, score = result

    with open(filename, "a") as fd:
        time = datetime.datetime.now()
        msg = "%s: %s at step %d: %f\n" % (time, metric, global_step, score)
        fd.write(msg)


def _save_validation_output(filename, decoded_symbols):
    # print validation output to file
    with open(filename, "w") as f:
        for symbols in decoded_symbols:
            f.write(" ".join(symbols) + "\n")


def _read_checkpoint_def(filename):
    records = []

    with tf.gfile.GFile(filename) as fd:
        fd.readline()

        for line in fd:
            records.append(line.strip().split(":")[-1].strip()[1:-1].split("/")[-1])

    return records


def _save_checkpoint_def(filename, checkpoint_names):
    keys = []

    for checkpoint_name in checkpoint_names:
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, checkpoint_name))

    sorted_names = sorted(keys, key=operator.itemgetter(0),
                          reverse=True)

    with tf.gfile.GFile(filename, "w") as fd:
        fd.write("model_checkpoint_path: \"%s\"\n" % checkpoint_names[0])

        for checkpoint_name in sorted_names:
            checkpoint_name = checkpoint_name[1]
            fd.write("all_model_checkpoint_paths: \"%s\"\n" % checkpoint_name)


def _read_score_record(filename):
    # "checkpoint_name": score
    records = []

    if not tf.gfile.Exists(filename):
        return records

    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            name, score = line.strip().split(":")
            name = name.strip()[1:-1]
            score = float(score)
            records.append([name, score])

    return records


def _save_score_record(filename, records):
    keys = []

    for record in records:
        checkpoint_name = record[0]
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, record))

    sorted_keys = sorted(keys, key=operator.itemgetter(0),
                         reverse=True)
    sorted_records = [item[1] for item in sorted_keys]

    with tf.gfile.GFile(filename, "w") as fd:
        for record in sorted_records:
            checkpoint_name, score = record
            fd.write("\"%s\": %f\n" % (checkpoint_name, score))


def _add_to_record(records, record, max_to_keep):
    added = None
    removed = None
    models = {}

    for (name, score) in records:
        models[name] = score

    if len(records) < max_to_keep:
        if record[0] not in models:
            added = record[0]
            records.append(record)
    else:
        sorted_records = sorted(records, key=lambda x: -x[1])
        worst_score = sorted_records[-1][1]
        current_score = record[1]

        if current_score >= worst_score:
            if record[0] not in models:
                added = record[0]
                removed = sorted_records[-1][0]
                records = sorted_records[:-1] + [record]

    # Sort
    records = sorted(records, key=lambda x: -x[1])

    return added, removed, records


def _evaluate(eval_fn, input_fn, decode_fn, path, config, placeholder):
    # placeholder指的是这里的placeholder是用句子作为source，还是用中间表示作为source，还是用bert的格式
    graph = tf.Graph()
    with graph.as_default():
        features = input_fn()
        refs = features["references"]
        if placeholder == "sentence":
            placeholders = {
                "source": tf.placeholder(tf.int32, [None, None], "source"),
                "source_length": tf.placeholder(tf.int32, [None], "source_length")
            }
        elif placeholder == "representation":
            placeholders = {
                "source": tf.placeholder(tf.float32, [None, None, 512], "source"),
                "source_length": tf.placeholder(tf.int32, [None], "source_length")
            }
        elif placeholder == "bert":  # bert!
            placeholders = {
                "input_ids": tf.placeholder(tf.int32, [None, None], "input_ids"),
                "source_length": tf.placeholder(tf.int32, [None], "source_length"),
                "input_type_ids": tf.placeholder(tf.int32, [None, None], "input_type_ids"),
                "input_mask": tf.placeholder(tf.int32, [None, None], "input_mask"),
            }
        else:
            raise ValueError("placeholder should either be 'sentence' or 'representation' or 'bert'.")
        predictions = eval_fn(placeholders)
        predictions = predictions[0][:, 0, :]

        all_refs = [[] for _ in range(len(refs))]
        all_outputs = []

        sess_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=path,
            config=config
        )

        with tf.train.MonitoredSession(session_creator=sess_creator) as sess:
            while not sess.should_stop():
                feats = sess.run(features)
                if placeholder == "bert":
                    outputs = sess.run(predictions, feed_dict={
                        placeholders["input_ids"]: feats["input_ids"],
                        placeholders["source_length"]: feats["source_length"],
                        placeholders["input_type_ids"]: feats["input_type_ids"],
                        placeholders["input_mask"]: feats["input_mask"],
                    })
                else:
                    outputs = sess.run(predictions, feed_dict={
                        placeholders["source"]: feats["source"],
                        placeholders["source_length"]: feats["source_length"]
                    })
                # shape: [batch, len]
                outputs = outputs.tolist()
                # shape: ([batch, len], ..., [batch, len])
                references = [item.tolist() for item in feats["references"]]

                all_outputs.extend(outputs)

                for i in range(len(refs)):
                    all_refs[i].extend(references[i])

        decoded_symbols = decode_fn(all_outputs)

        for i, l in enumerate(decoded_symbols):
            decoded_symbols[i] = " ".join(l).replace("@@ ", "").split()

        decoded_refs = [decode_fn(refs) for refs in all_refs]
        decoded_refs = [list(x) for x in zip(*decoded_refs)]

        # return decoded_symbols to debug
        return decoded_symbols, bleu.bleu(decoded_symbols, decoded_refs)


class EvaluationHook(tf.train.SessionRunHook):
    """ Validate and save checkpoints every N steps or seconds.
        This hook only saves checkpoint according to a specific metric.
    """

    def __init__(self, eval_fn, eval_input_fn, eval_decode_fn, base_dir,
                 session_config, max_to_keep=5, eval_secs=None,
                 eval_steps=None, metric="BLEU", placeholder="sentence"):
        """ Initializes a `EvaluationHook`.
        :param eval_fn: A function with signature (feature)
        :param eval_input_fn: A function with signature ()
        :param eval_decode_fn: A function with signature (inputs)
        :param base_dir: A string. Base directory for the checkpoint files.
        :param session_config: An instance of tf.ConfigProto
        :param max_to_keep: An integer. The maximum of checkpoints to save
        :param eval_secs: An integer, eval every N secs.
        :param eval_steps: An integer, eval every N steps.
        :param checkpoint_basename: `str`, base name for the checkpoint files.
        :param placeholder: what placeholder should the _evaluate function use.
                "sentence", "representation" or "bert".
        :raises ValueError: One of `save_steps` or `save_secs` should be set.
        :raises ValueError: At most one of saver or scaffold should be set.
        """
        tf.logging.info("Create EvaluationHook.")

        if metric != "BLEU":
            raise ValueError("Currently, EvaluationHook only support BLEU")

        self._base_dir = base_dir.rstrip("/")
        self._debug_dir = os.path.join(base_dir, "debug")  # debug
        self._session_config = session_config
        self._save_path = os.path.join(base_dir, "eval")
        self._record_name = os.path.join(self._save_path, "record")
        self._log_name = os.path.join(self._save_path, "log")
        self._eval_fn = eval_fn
        self._eval_input_fn = eval_input_fn
        self._eval_decode_fn = eval_decode_fn
        self._max_to_keep = max_to_keep
        self._metric = metric
        self._global_step = None
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=eval_secs or None, every_steps=eval_steps or None
        )
        self._placeholder = placeholder

    def begin(self):
        if self._timer.last_triggered_step() is None:
            self._timer.update_last_triggered_step(0)

        global_step = tf.train.get_global_step()

        if not tf.gfile.Exists(self._save_path):
            tf.logging.info("Making dir: %s" % self._save_path)
            tf.gfile.MakeDirs(self._save_path)

        # debug dir
        if not tf.gfile.Exists(self._debug_dir):
            tf.logging.info("Making dir: %s" % self._debug_dir)
            tf.gfile.MakeDirs(self._debug_dir)

        params_pattern = os.path.join(self._base_dir, "*.json")
        params_files = tf.gfile.Glob(params_pattern)

        for name in params_files:
            new_name = name.replace(self._base_dir, self._save_path)
            tf.gfile.Copy(name, new_name, overwrite=True)

        if global_step is None:
            raise RuntimeError("Global step should be created first")

        self._global_step = global_step

    def before_run(self, run_context):
        args = tf.train.SessionRunArgs(self._global_step)
        return args

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results

        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step)

            # Get the real value
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                # Save model
                save_path = os.path.join(self._base_dir, "model.ckpt")
                saver = _get_saver()
                tf.logging.info("Saving checkpoints for %d into %s." %
                                (global_step, save_path))
                saver.save(run_context.session,
                           save_path,
                           global_step=global_step)
                # Do validation here
                # Debug: print validation result
                tf.logging.info("Validating model at step %d" % global_step)
                decoded_symbols, score = _evaluate(self._eval_fn, self._eval_input_fn,
                                  self._eval_decode_fn,
                                  self._base_dir,
                                  self._session_config,
                                  self._placeholder)

                tf.logging.info("%s at step %d: %f" %
                                (self._metric, global_step, score))

                _save_log(self._log_name, (self._metric, global_step, score))
                _save_validation_output(os.path.join(self._debug_dir, "%d.out" % global_step), decoded_symbols)

                checkpoint_filename = os.path.join(self._base_dir,
                                                   "checkpoint")
                all_checkpoints = _read_checkpoint_def(checkpoint_filename)
                records = _read_score_record(self._record_name)
                latest_checkpoint = all_checkpoints[-1]
                record = [latest_checkpoint, score]
                added, removed, records = _add_to_record(records, record,
                                                         self._max_to_keep)

                if added is not None:
                    old_path = os.path.join(self._base_dir, added)
                    new_path = os.path.join(self._save_path, added)
                    old_files = tf.gfile.Glob(old_path + "*")
                    tf.logging.info("Copying %s to %s" % (old_path, new_path))

                    for o_file in old_files:
                        n_file = o_file.replace(old_path, new_path)
                        tf.gfile.Copy(o_file, n_file, overwrite=True)

                if removed is not None:
                    filename = os.path.join(self._save_path, removed)
                    tf.logging.info("Removing %s" % filename)
                    files = tf.gfile.Glob(filename + "*")

                    for name in files:
                        tf.gfile.Remove(name)

                _save_score_record(self._record_name, records)
                checkpoint_filename = checkpoint_filename.replace(
                    self._base_dir, self._save_path
                )
                _save_checkpoint_def(checkpoint_filename,
                                     [item[0] for item in records])

                best_score = records[0][1]
                tf.logging.info("Best score at step %d: %f" %
                                (global_step, best_score))

    def end(self, session):
        last_step = session.run(self._global_step)

        if last_step != self._timer.last_triggered_step():
            global_step = last_step
            tf.logging.info("Validating model at step %d" % global_step)
            score = _evaluate(self._eval_fn, self._eval_input_fn,
                              self._eval_decode_fn,
                              self._base_dir,
                              self._session_config,
                              self._placeholder)
            tf.logging.info("%s at step %d: %f" %
                            (self._metric, global_step, score))

            checkpoint_filename = os.path.join(self._base_dir,
                                               "checkpoint")
            all_checkpoints = _read_checkpoint_def(checkpoint_filename)
            records = _read_score_record(self._record_name)
            latest_checkpoint = all_checkpoints[-1]
            record = [latest_checkpoint, score]
            added, removed, records = _add_to_record(records, record,
                                                     self._max_to_keep)

            if added is not None:
                old_path = os.path.join(self._base_dir, added)
                new_path = os.path.join(self._save_path, added)
                old_files = tf.gfile.Glob(old_path + "*")
                tf.logging.info("Copying %s to %s" % (old_path, new_path))

                for o_file in old_files:
                    n_file = o_file.replace(old_path, new_path)
                    tf.gfile.Copy(o_file, n_file, overwrite=True)

            if removed is not None:
                filename = os.path.join(self._save_path, removed)
                tf.logging.info("Removing %s" % filename)
                files = tf.gfile.Glob(filename + "*")

                for name in files:
                    tf.gfile.Remove(name)

            _save_score_record(self._record_name, records)
            checkpoint_filename = checkpoint_filename.replace(
                self._base_dir, self._save_path
            )
            _save_checkpoint_def(checkpoint_filename,
                                 [item[0] for item in records])

            best_score = records[0][1]
            tf.logging.info("Best score: %f" % best_score)


class MultiStepHook(tf.train.SessionRunHook):

    def __init__(self, hook, step=1):
        self._hook = hook
        self._step = step
        self._iter = 0 if step == 1 else 1

    def begin(self):
        self._hook.begin()

    def after_create_session(self, session, coord):
        self._hook.after_create_session(session, coord)

    def before_run(self, run_context):
        return self._hook.before_run(run_context)

    def after_run(self, run_context, run_values):
        if self._iter % self._step == 0:
            self._hook.after_run(run_context, run_values)
        self._iter = (self._iter + 1) % self._step

    def end(self, session):
        self._hook.end(session)
