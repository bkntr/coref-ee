#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import util
import coref_model as cm

if __name__ == "__main__":
  args = util.get_args()
  config = util.initialize_from_env(args.experiment, args.logdir)
  config["eval_path"] = "test.english.jsonlines"
  config["conll_eval_path"] = "test.english.v4_gold_conll"
  config["context_embeddings"]["path"] = "glove.840B.300d.txt"

  model = cm.CorefModel(config, eval_mode=True)
  with tf.Session() as session:
    model.restore(session, args.latest_checkpoint)
    model.evaluate(session, official_stdout=True, pprint=False, test=True)
