# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

import numpy as np
import tensorflow as tf
import h5py

import modeling
import tokenization
from data import process_example
from tqdm import tqdm

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_integer(
    "window_size", 511,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "stride", 127,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


def _convert_example_to_features(example, window_start, window_end, tokens_ids_to_extract, tokenizer, seq_length):
    window_tokens = example.tokens[window_start:window_end]

    tokens = []
    segment_ids = []
    for token in window_tokens:
        tokens.append(token)
        segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    extract_mask = [0] * seq_length
    for i in tokens_ids_to_extract:
        extract_mask[i - window_start] = 1

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(segment_ids) == seq_length

    return dict(doc_index=example.document_index,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                extract_mask=extract_mask,
                tokens=tokens)


def convert_examples_to_features(bert_examples, orig_examples, window_size, stride, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    assert window_size % 2 == 1
    assert stride % 2 == 1

    for bert_example, orig_example in zip(bert_examples, orig_examples):
        current_example_features = []
        for i in range(0, len(bert_example.tokens), stride):
            window_center = i + window_size // 2
            token_ids_to_extract = []
            extract_start = int(np.clip(window_center - stride // 2, 0, len(bert_example.tokens)))
            extract_end = int(np.clip(window_center + stride // 2 + 1, extract_start, len(bert_example.tokens)))

            if i == 0:
                token_ids_to_extract.extend(range(extract_start))

            token_ids_to_extract.extend(range(extract_start, extract_end))

            if i + window_size >= len(bert_example.tokens):
                token_ids_to_extract.extend(range(extract_end, len(bert_example.tokens)))

            token_ids_to_extract = [t for t in token_ids_to_extract if bert_example.bert_to_orig_map[t] >= 0]

            features = _convert_example_to_features(bert_example,
                                                    i,
                                                    min(i + window_size, len(bert_example.tokens)),
                                                    token_ids_to_extract,
                                                    tokenizer,
                                                    window_size)

            current_example_features.append(features)

            if i + window_size >= len(bert_example.tokens):
                break

        current_example_features = {k: np.array([c[k] for c in current_example_features]) for k in current_example_features[0]}

        max_sentence_len = max(len(s) for s in orig_example.sentence_tokens)
        extract_sentences = np.zeros((len(orig_example.sentence_tokens), max_sentence_len), np.int32)
        extract_mask = current_example_features['extract_mask']
        extract_idxs = extract_mask.cumsum().reshape(extract_mask.shape)
        for c in range(extract_mask.shape[0]):
            for i in range(extract_mask.shape[1]):
                if extract_mask[c, i]:
                    si, sj = orig_example.unravel_token_index(extract_idxs[c, i] - 1)
                    extract_sentences[si, sj] = c * extract_mask.shape[1] + i + 1
        current_example_features['extract_sentences'] = extract_sentences

        yield current_example_features


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=False)

    json_examples = []
    for x in ['test', 'train', 'dev']:
    # for x in ['test']:
        with open(os.path.join(FLAGS.input_file, x + '.english.jsonlines')) as f:
            json_examples.extend((json.loads(jsonline) for jsonline in f.readlines()))

    orig_examples = []
    bert_examples = []
    for i, json_e in enumerate(json_examples):
        e = process_example(json_e, i, should_filter_embedded_mentions=True)
        orig_examples.append(e)
        bert_examples.append(e.bertify(tokenizer))

    writer = h5py.File(FLAGS.output_file, 'w')
    for data in tqdm(convert_examples_to_features(bert_examples,
                                                  orig_examples,
                                                  FLAGS.window_size,
                                                  FLAGS.stride,
                                                  tokenizer), total=len(json_examples)):
        document_index = int(data["doc_index"][0])
        bert_example = bert_examples[document_index]
        dataset_key = bert_example.doc_key.replace('/', ':')

        sentences = []
        for sentence_indices in data['extract_sentences']:
            cur_sentence = []
            for i in sentence_indices:
                tokens_flattened = sum([list(ts) for ts in data['tokens']], [])
                if i > 0:
                    cur_sentence.append(tokens_flattened[i - 1])
            sentences.append(cur_sentence)
        assert [len(s) for s in sentences] == [len(s) for s in orig_examples[document_index].sentence_tokens]
        sentences_flattened = sum(sentences, [])
        expected = [t for i, t in enumerate(bert_example.tokens) if bert_example.bert_to_orig_map[i] >= 0]
        assert sentences_flattened == expected

        writer.create_dataset('{}/input_ids'.format(dataset_key), data=data['input_ids'])
        writer.create_dataset('{}/input_mask'.format(dataset_key), data=data['input_mask'])
        writer.create_dataset('{}/segment_ids'.format(dataset_key), data=data['segment_ids'])
        writer.create_dataset('{}/extract_mask'.format(dataset_key), data=data['extract_mask'])
        writer.create_dataset('{}/extract_sentences'.format(dataset_key), data=data['extract_sentences'])
        # for i, s in enumerate(data['tokens']):
        #     tokens_dset = writer.create_dataset('{}/tokens/{}'.format(dataset_key, i),
        #                                         (len(s),),
        #                                         dtype=h5py.special_dtype(vlen=unicode))
        #     for j, w in enumerate(s):
        #         tokens_dset[j] = w
    writer.close()

    # dataset = tf.data.Dataset.from_generator(functools.partial(convert_examples_to_features,
    #                                                            bert_examples,
    #                                                            FLAGS.window_size,
    #                                                            FLAGS.stride,
    #                                                            tokenizer),
    #                                          dict(doc_index=tf.int32,
    #                                               input_ids=tf.int32,
    #                                               input_mask=tf.int32,
    #                                               segment_ids=tf.int32,
    #                                               extract_mask=tf.int32),
    #                                          dict(doc_index=tf.TensorShape([None]),
    #                                               input_ids=tf.TensorShape([None, FLAGS.window_size]),
    #                                               input_mask=tf.TensorShape([None, FLAGS.window_size]),
    #                                               segment_ids=tf.TensorShape([None, FLAGS.window_size]),
    #                                               extract_mask=tf.TensorShape([None, FLAGS.window_size]))
    #                                          )
    # inputs = dataset.make_one_shot_iterator().get_next()
    #
    # bert_inputs = dict(
    #     input_ids=tf.expand_dims(inputs['input_ids'], 1),
    #     input_mask=tf.expand_dims(inputs['input_mask'], 1),
    #     segment_ids=tf.expand_dims(inputs['segment_ids'], 1))
    #
    # bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    # bert_outputs = tf.map_fn(
    #     lambda x: modeling.BertModel(config=bert_config,
    #                                  is_training=True,
    #                                  input_ids=x['input_ids'],
    #                                  input_mask=x['input_mask'],
    #                                  token_type_ids=x['segment_ids'],
    #                                  use_one_hot_embeddings=FLAGS.use_one_hot_embeddings).sequence_output,
    #     bert_inputs, dtype=tf.float32, parallel_iterations=1, swap_memory=True)
    # # repeat = 5
    # # bert_outputs = modeling.BertModel(config=bert_config,
    # #                                   is_training=True,
    # #                                   input_ids=tf.tile(inputs['input_ids'], [repeat, 1]),
    # #                                   input_mask=tf.tile(inputs['input_mask'], [repeat, 1]),
    # #                                   token_type_ids=tf.tile(inputs['segment_ids'], [repeat, 1]),
    # #                                   use_one_hot_embeddings=FLAGS.use_one_hot_embeddings).sequence_output
    # loss = tf.nn.sigmoid(tf.reduce_mean(bert_outputs))
    # train_op = tf.train.AdamOptimizer().minimize(loss)
    # # bert_outputs_masked = tf.squeeze(bert_outputs, 1)
    # # bert_outputs_masked = tf.boolean_mask(bert_outputs_masked, inputs['extract_mask'])
    #
    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     while True:
    #         # features_raw, features, doc_index = sess.run([bert_outputs, bert_outputs_masked, inputs['doc_index']])
    #         # print(len(features_raw), features.shape, len(bert_examples[doc_index[0]].tokens))
    #         l, _ = sess.run([loss, train_op])
    #         print(l)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_file")
    tf.app.run()
