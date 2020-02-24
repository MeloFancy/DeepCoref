import logging
import os

import math
from transformers import is_tf_available

from data_prepare.prepare_data import get_term_embedding
from processor import CorefProcessor
from utils_ import DataProcessor, InputExample, InputFeatures

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

def convert_examples_to_features(examples, tokenizer, embedding, unk_token,
                                      max_length=512,
                                      # task=None,
                                      label_list=None,
                                      output_mode=None,
                                      # pad_on_left=False,
                                      # pad_token=0,
                                      # pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    # print(task)
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    # if task is not None:
    processor = CorefProcessor()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s" % label_list)
    if output_mode is None:
        output_mode = "classification"
        logger.info("Using output mode %s" % output_mode)

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        # inputs = tokenizer.encode_plus(
        #     example.text_a,
        #     example.text_b,
        #     add_special_tokens=True,
        #     max_length=max_length,
        # )
        # input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        #
        # # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # # tokens are attended to.
        # attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        #
        # # Zero-pad up to the sequence length.
        # padding_length = max_length - len(input_ids)
        # if pad_on_left:
        #     input_ids = ([pad_token] * padding_length) + input_ids
        #     attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        #     token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        # else:
        #     input_ids = input_ids + ([pad_token] * padding_length)
        #     attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        #     token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        feature = create_features(example, tokenizer, max_length)
        input_ids, token_type_ids, attention_mask, term_mask = feature["input_ids"], feature["token_type_ids"], feature["attention_mask"], feature["term_mask"]

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
        assert len(term_mask) == max_length, "Error with input length {} vs {}".format(len(term_mask), max_length)

        # assert input_ids == feature['input_ids']
        # assert token_type_ids == feature['token_type_ids']
        # assert attention_mask == feature['attention_mask']

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("original text: %s" % example.text_a + ' ' + example.text_b)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

            # print(feature)

        veca = get_term_embedding(example.term_a, embedding, unk_token)
        vecb = get_term_embedding(example.term_b, embedding, unk_token)

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              term_mask=term_mask,
                              terma_vec=veca,
                              termb_vec=vecb,
                              label=label))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features


def create_features(example, tokenizer, max_seq_length):

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    texta_length = int(max_num_tokens / 2)
    textb_length = max_num_tokens - texta_length

    # texta_tokens, terma_tokens = example.text_a.split(), example.term_a.split()
    # textb_tokens, termb_tokens = example.text_b.split(), example.term_b.split()

    # starta_index, enda_index = index_pos(texta_tokens, terma_tokens)
    # assert ' '.join(texta_tokens[starta_index : enda_index + 1]) == example.term_a
    # startb_index, endb_index = index_pos(textb_tokens, termb_tokens)
    # assert ' '.join(textb_tokens[startb_index: endb_index + 1]) == example.term_b

    # texta_context, terma_start, terma_end = get_context_tokens(
    #     texta_tokens, starta_index, enda_index, texta_length, tokenizer)
    # textb_context, termb_start, termb_end = get_context_tokens(
    #     textb_tokens, startb_index, endb_index, textb_length, tokenizer)

    # print(texta_context, terma_start, terma_end)
    # print(textb_context, termb_start, termb_end)

    feature = {}

    # tokens_a, tokens_b = texta_context, textb_context
    tokens_a, tokens_b = tokenizer.tokenize(example.text_a)[:texta_length], tokenizer.tokenize(example.text_b)[:textb_length]
    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_type_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b) + 1)
    attention_mask = [1]*len(input_ids)
    term_mask = [0]*len(input_ids)

    # Update these indices to take [CLS] into account
    # new_terma_start, new_terma_end = terma_start + 1, terma_end + 1
    # new_termb_start, new_termb_end = termb_start + (len(tokens_a) + 2), termb_end + (len(tokens_a) + 2)

    # assert tokens[new_terma_start: new_terma_end+1] == tokenizer.tokenize(example.term_a)
    # assert tokens[new_termb_start: new_termb_end+1] == tokenizer.tokenize(example.term_b)

    # for t in range(new_terma_start, new_terma_end+1):
    #     term_mask[t] = 1
    # for t in range(new_termb_start, new_termb_end+1):
    #     term_mask[t] = 1

    assert len(input_ids) <= max_seq_length

    tokens = tokens + ['<pad>'] * (max_seq_length - len(tokens))
    feature['tokens'] = tokens
    feature['input_ids'] = pad_sequence(input_ids, max_seq_length)
    feature['token_type_ids'] = pad_sequence(token_type_ids, max_seq_length)
    feature['attention_mask'] = pad_sequence(attention_mask, max_seq_length)
    feature['term_mask'] = pad_sequence(term_mask, max_seq_length)

    return feature


def get_context_tokens(context_tokens, start_index, end_index, max_tokens, tokenizer):
    start_pos = start_index - max_tokens
    if start_pos < 0:
        start_pos = 0
    prefix = ' '.join(context_tokens[start_pos: start_index])
    suffix = ' '.join(context_tokens[end_index+1: end_index+max_tokens+1])

    prefix = tokenizer.tokenize(prefix)
    suffix = tokenizer.tokenize(suffix)
    mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index+1]))

    assert len(mention) < max_tokens

    remaining_tokens = max_tokens - len(mention)
    half_remaining_tokens = int(math.ceil(1.0*remaining_tokens/2))

    mention_context = []

    if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
        prefix_len = half_remaining_tokens
    elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
        prefix_len = remaining_tokens - len(suffix)
    elif len(prefix) < half_remaining_tokens:
        prefix_len = len(prefix)

    if prefix_len > len(prefix):
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]

    mention_context = prefix + mention + suffix
    mention_start = len(prefix)
    mention_end = mention_start + len(mention) - 1
    mention_context = mention_context[:max_tokens]

    assert mention_start <= max_tokens
    assert mention_end <= max_tokens

    return mention_context, mention_start, mention_end


def index_pos(text_tokens, term_tokens):
    term_length = len(term_tokens)
    start_index = 0
    for index, token in enumerate(text_tokens):
        if ' '.join(text_tokens[index:index + term_length]) == ' '.join(term_tokens):
            start_index = index
            break
    end_index = start_index + term_length - 1
    return start_index, end_index


def pad_sequence(tokens, max_len):
    assert len(tokens) <= max_len
    return tokens + [0]*(max_len - len(tokens))
