# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.rnnsearch_lrp
import thumt.models.transformer
import thumt.models.transformer_lrp
import thumt.models.transformer_frozen
import thumt.models.transformer_adapt
import thumt.models.transformer_bert_encoder


def get_model(name, lrp=False, frozen=False, adapt=False):
    name = name.lower()

    if name == "rnnsearch":
        if not lrp:
            return thumt.models.rnnsearch.RNNsearch
        else:
            return thumt.models.rnnsearch_lrp.RNNsearchLRP
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name == "transformer":
        if not lrp and not frozen and not adapt:
            return thumt.models.transformer.Transformer
        elif frozen and not adapt:
            return thumt.models.transformer_frozen.Transformer
        elif adapt:
            return thumt.models.transformer_adapt.Transformer
        else:
            return thumt.models.transformer_lrp.TransformerLRP
    elif name == "bert-transformer":
        return thumt.models.transformer_bert_encoder.Transformer
    else:
        raise LookupError("Unknown model %s" % name)
