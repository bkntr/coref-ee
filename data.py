from collections import defaultdict

import numpy as np
import random

import tokenization


class Example(object):
    def __init__(self, doc_key, tokens, sentence_tokens, gold_starts, gold_ends, speaker_ids, cluster_ids, genre, document_index,
                 offset=0, bert_to_orig_map=None):
        assert len(tokens) == len(speaker_ids)

        self.doc_key = doc_key
        self.tokens = tokens
        self.sentence_tokens = sentence_tokens
        self.gold_starts = gold_starts
        self.gold_ends = gold_ends
        self.speaker_ids = speaker_ids
        self.cluster_ids = cluster_ids
        self.genre = genre
        self.document_index = document_index
        self.offset = offset
        self.bert_to_orig_map = bert_to_orig_map

    def truncate(self, start, size):
        # don't truncate in the middle of a mention
        for mention in zip(self.gold_starts, self.gold_ends):
            if index_in_mention(start, mention):
                start = mention[0]

            if index_in_mention(start + size, mention):
                size -= start + size - mention[0]
        end = start + size

        tokens = self.tokens[start:end]
        sentence_tokens = None
        speaker_ids = self.speaker_ids[start:end]
        gold_spans = np.logical_and(self.gold_starts >= start, self.gold_ends < end)
        gold_starts = self.gold_starts[gold_spans] - start
        gold_ends = self.gold_ends[gold_spans] - start
        cluster_ids = self.cluster_ids[gold_spans]

        return Example(self.doc_key, tokens, sentence_tokens, gold_starts, gold_ends, speaker_ids, cluster_ids,
                       self.genre, self.document_index, start)

    def bertify(self, tokenizer):
        assert self.offset == 0

        bert_tokens = []
        orig_to_bert_map = []
        orig_to_bert_end_map = []
        bert_speaker_ids = []
        for t, s in zip(self.tokens, self.speaker_ids):
            bert_t = tokenizer.tokenize(t)
            orig_to_bert_map.append(len(bert_tokens))
            orig_to_bert_end_map.append(len(bert_tokens) + len(bert_t) - 1)
            bert_tokens.extend(bert_t)
            bert_speaker_ids.extend([s] * len(bert_t))

        bert_sentence_tokens = [tokenizer.tokenize(' '.join(s)) for s in self.sentence_tokens]

        bert_to_orig_map = [-1] * len(bert_tokens)
        for i, bert_i in enumerate(orig_to_bert_map):
            bert_to_orig_map[bert_i] = i

        orig_to_bert_map = np.array(orig_to_bert_map)
        orig_to_bert_end_map = np.array(orig_to_bert_end_map)
        if len(self.gold_starts):
            gold_starts = orig_to_bert_map[self.gold_starts]
            gold_ends = orig_to_bert_end_map[self.gold_ends]
        else:
            gold_starts = self.gold_starts
            gold_ends = self.gold_ends

        return Example(self.doc_key, bert_tokens, bert_sentence_tokens, gold_starts, gold_ends, bert_speaker_ids,
                       self.cluster_ids, self.genre, self.document_index, bert_to_orig_map=bert_to_orig_map)

    def unravel_token_index(self, token_index):
        prev_sentences_len = 0
        for i, s in enumerate(self.sentence_tokens):
            if token_index < prev_sentences_len + len(s):
                token_index_in_sentence = token_index - prev_sentences_len
                return i, token_index_in_sentence
            prev_sentences_len += len(s)

        raise ValueError('token_index is out of range ({} >= {})', token_index, len(self.tokens))


def index_in_mention(index, mention):
    return mention[0] <= index and mention[1] >= index


def mention_contains(mention1, mention2):
    return mention1[0] <= mention2[0] and mention1[1] >= mention2[1]


def filter_embedded_mentions(mentions):
    """
    Filter out mentions embedded in other mentions
    """
    filtered = []
    for i, m in enumerate(mentions):
        other_mentions = mentions[:i] + mentions[i + 1:]
        if any(mention_contains(other_m, m) for other_m in other_mentions):
            continue
        filtered.append(m)
    return filtered


def filter_overlapping_mentions(mentions):
    start_to_mentions = defaultdict(list)
    for m in mentions:
        start_to_mentions[m[0]].append(m)

    filtered_mentions = []
    for ms in start_to_mentions.values():
        if len(ms) > 1:
            pass
        max_mention = np.argmax([m[1] - m[0] for m in ms])
        filtered_mentions.append(ms[max_mention])

    return filtered_mentions


def flatten(l):
    return [item for sublist in l for item in sublist]


def tensorize_mentions(mentions):
    if len(mentions) > 0:
        starts, ends = zip(*mentions)
    else:
        starts, ends = [], []
    return np.array(starts), np.array(ends)


genres = {g: i for i, g in enumerate(["bc", "bn", "mz", "nw", "pt", "tc", "wb"])}

def process_example(example, index, should_filter_embedded_mentions=False):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in flatten(clusters))
    if should_filter_embedded_mentions:
        gold_mentions = filter_overlapping_mentions(gold_mentions)
        # gold_mentions = filter_embedded_mentions(gold_mentions)
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            if tuple(mention) in gold_mention_map:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = flatten(example["speakers"])

    assert num_words == len(speakers)

    sentence_tokens = [[tokenization.convert_to_unicode(w) for w in s] for s in sentences]

    tokens = sum(sentence_tokens, [])

    speaker_dict = {s: i for i, s in enumerate(set(speakers))}
    speaker_ids = np.array([speaker_dict[s] for s in speakers])

    # TODO: genre
    doc_key = example["doc_key"]
    genre = genres[doc_key[:2]]

    gold_starts, gold_ends = tensorize_mentions(sorted(gold_mentions))

    return Example(doc_key, tokens, sentence_tokens, gold_starts, gold_ends, speaker_ids, cluster_ids, genre, index)
