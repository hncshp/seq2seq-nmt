# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Author: Huang Ping
# Date: June 28, 2018
#
#                                          ****  MODEL DESCRIPTION  ****
#                        Bi-directional LSTM in Encoder, Multi Uni-directional LSTM in Decoder.
# ----------------------------------------------------------------------------------------------------------------------

from __future__ import division

from six.moves import range
from tensorflow.python.estimator.model_fn import ModeKeys
import argparse
import sys
import numpy as np
import tensorflow as tf
import random
from tqdm import trange
from tensorflow.python.ops import lookup_ops
import math
import logging
from time import time
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.DEBUG and tf.logging.INFO)
def log(msg, level=logging.INFO):
    tf.logging.log(level, msg)
def get_time():
    return time()

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------Global parameters definition-------------------------------------------------
w_split = True

if w_split:
    PAD = '<PAD>'
    EOS = '</S>'
    UNK = '<UNK>'
    data_dir = './en_vi/'
else:
    PAD = '^'
    EOS = '#'
    UNK = '$'
    data_dir = './data/'

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

FLAGS = None
encoder_word_int_map, encoder_vocab, encoder_vocab_size = None, None, None
decoder_word_int_map, decoder_vocab, decoder_vocab_size = None, None, None

src_bucket_width, tgt_bucket_width = None, None
src_max_length, tgt_max_length = 0, 0


# -----------------------------------------Global parameters definition-------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------functions definition for whole program utility----------------------------------------

def create_vocab_tables(src_vocab_file, tgt_vocab_file):
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)

    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID)

    return src_vocab_table, tgt_vocab_table

def np_vocab_processing(vocab_file):
    vocab = []
    vocab_size = 0
    with open(vocab_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            vocab_size += 1
            word = line.strip()
            vocab.append(word)
    word_2_int_map = {word: index for index, word in enumerate(vocab)}
    return word_2_int_map, vocab, vocab_size


def max_line_length(file_name):
    max_length = 0
    with open(file_name, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            if len(content) > max_length:
                max_length = len(content)
    return max_length


def average_length(file_name):
    total_length = 0
    item_num = 0
    with open(file_name, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            if len(content) > 0:
                content += [EOS]
                item_num += 1
                total_length += len(content)

    return round((total_length / item_num) * FLAGS.alpha)

def one_hot(one_data, vocab_size):
    length = len(one_data)
    one_hot_array = np.zeros(shape=(length, vocab_size), dtype=np.float32)
    for time in range(length):
        one_hot_array[time][one_data[time]] = 1.0
    return one_hot_array

def en_translation_file_processing(translation_file, seq_length, vocab_size):
    outcome = []
    with open(translation_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            feature_vector = list(map(lambda word: encoder_word_int_map.get(word, UNK_ID), content))
            feature_temp = np.full((seq_length), PAD_ID, np.int32)  # full fill with PAD_ID
            if len(feature_vector) > (seq_length - 1):
                feature_temp[:(seq_length - 1)] = feature_vector[:(seq_length - 1)]
                feature_temp[(seq_length - 1)] = EOS_ID
            else:
                feature_temp[:len(feature_vector)] = feature_vector
                feature_temp[len(feature_vector)] = EOS_ID
            feature_one_hot = one_hot(feature_temp, vocab_size)
            outcome.append(feature_one_hot)  # [B,T,V]
    return outcome

def de_translation_file_processing(translation_file, seq_length, vocab_size):
    outcome = []
    with open(translation_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            if FLAGS.whitespace_or_nonws_slip:
                content = line.strip().split()
            else:
                content = line.strip()
            feature_vector = list(map(lambda word: decoder_word_int_map.get(word, UNK_ID), content))
            feature_temp = np.full((seq_length), PAD_ID, np.int32)  # full fill with PAD_ID
            if len(feature_vector) > (seq_length - 1):
                feature_temp[:(seq_length - 1)] = feature_vector[:(seq_length - 1)]
                feature_temp[(seq_length - 1)] = EOS_ID
            else:
                feature_temp[:len(feature_vector)] = feature_vector
                feature_temp[len(feature_vector)] = EOS_ID
            feature_one_hot = one_hot(feature_temp, vocab_size)
            outcome.append(feature_one_hot)  # [B,T,V]
    return outcome

def en_characters(probabilities):
    return [encoder_vocab[c] for c in np.argmax(probabilities, 1)]

def en_batches2string(batches):
    s = [''] * batches[0].shape[0]
    for b in batches:
        if FLAGS.whitespace_or_nonws_slip:
            s = [' '.join(x) for x in zip(s, en_characters(b))]
        else:
            s = [''.join(x) for x in zip(s, en_characters(b))]

    return s


def de_characters(probabilities):
    return [decoder_vocab[c] for c in np.argmax(probabilities, 1)]

def de_batches2string(batches):
    s = [''] * batches[0].shape[0]
    for b in batches:
        if FLAGS.whitespace_or_nonws_slip:
            s = [' '.join(x) for x in zip(s, de_characters(b))]
        else:
            s = [''.join(x) for x in zip(s, de_characters(b))]
    return s

def accuracy(labels, predictions):
    return np.sum(np.argmax(labels, axis=-1) == np.argmax(predictions, axis=-1)) / (labels.shape[0] * labels.shape[1])


# --------------------------------functions definition for whole program utility----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------define the FLAGS parameters for whole model usage.---------------------------------------

def add_Argumets(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--hard_att_model_dir", type=str, default= data_dir+'hard_model_dir',
                        help='Output directory for hard attention model and training stats.')
    parser.add_argument("--soft_att_model_dir", type=str, default= data_dir+'soft_model_dir',
                        help='Output directory for soft attention model and training stats.')
    parser.add_argument("--train_src_file_path", type=str, default= data_dir + 'train_src.txt',
                        help='training src data path.')
    parser.add_argument("--train_tgt_file_path", type=str, default= data_dir + 'train_tgt.txt',
                        help='training tgt data path.')
    parser.add_argument("--eval_src_file_path", type=str, default= data_dir + 'eval_src.txt',
                        help='eval src data path.')
    parser.add_argument("--eval_tgt_file_path", type=str, default= data_dir + 'eval_tgt.txt',
                        help='eval tgt data path.')
    parser.add_argument("--infer_src_file_path", type=str, default= data_dir + 'infer_src.txt',
                        help='infer src data path.')
    parser.add_argument("--infer_tgt_file_path", type=str, default= data_dir + 'infer_tgt.txt',
                        help='infer tgt data path.')
    parser.add_argument("--src_vocab_file_path", type=str, default= data_dir + 'vocab_src.txt',
                        help='src vocab path.')
    parser.add_argument("--tgt_vocab_file_path", type=str, default= data_dir + 'vocab_tgt.txt',
                        help='tgt vocab path.')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='batch size.')
    parser.add_argument("--alpha", type=float, default=1.0,
                        help='average length multiplier ratio.')
    parser.add_argument("--input_vocab_size", type=int, default=0,
                        help='input vocabulary size.')
    parser.add_argument("--output_vocab_size", type=int, default=0,
                        help='output vocabulary size.')
    parser.add_argument("--max_input_time_steps", type=int, default=50,
                        help='input sequence length.')
    parser.add_argument("--max_output_time_steps", type=int, default=60,
                        help='output sequence length.')
    parser.add_argument("--infer_max_input_time_steps", type=int, default=0,
                        help='input sequence length.')
    parser.add_argument("--infer_max_output_time_steps", type=int, default=0,
                        help='output sequence length.')
    parser.add_argument("--num_buckets", type=int, default=5,
                        help='number of buckets.')
    parser.add_argument("--src_bucket_width", type=int, default=0,
                        help='src_bucket_width.')
    parser.add_argument("--tgt_bucket_width", type=int, default=0,
                        help='tgt_bucket_width.')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--num_units", type=int, default=512,
                        help="hidden node number.")
    # ------------------------------------------------------------------------------------------------------------------
    # num_layers and bi_or_uni shall be configured together.
    # if num_layers == 1, then bi_or_uni == False, mean uni-direction.
    # if num_layers >= 2, then bi_or_uni == True or False.
    parser.add_argument("--num_layers", type=int, default=4,
                        help="layer number.")
    parser.add_argument("--bi_or_uni", type="bool", nargs='?', const=True, default=True,
                        help="True means bi-direction; False means uni-direction")
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate: 1.0 for sgd; Adam: 0.001 | 0.0001")
    parser.add_argument("--decay_factor", type=float, default=0.5,
                        help="Learning rate decay_factor")
    parser.add_argument("--reg_lambda", type=float, default=0.001,
                        help="L1 or L2 regularization lambda")
    parser.add_argument("--train_steps", type=int, default=50000,
                        help="total training steps")
    parser.add_argument("--warmup_steps", type=int, default=50000,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="max_gradient_norm.")
    parser.add_argument("--predict_window", type=int, default=5,
                        help="windows size in hard attention mode. so 2*win_size+1 steps are considered in")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="beam search width.")
    parser.add_argument("--lr_warmup", type="bool", nargs='?', const=True, default=False,
                        help="True means use learning rate warmup schema")
    parser.add_argument("--lr_decay", type="bool", nargs='?', const=True, default=False,
                        help="True means use learning rate decay schema")
    parser.add_argument("--whitespace_or_nonws_slip", type="bool", nargs='?', const=True, default=w_split,
                        help="True means whitespace slip; False means no whitespace slip")
    parser.add_argument("--fix_or_variable_length", type="bool", nargs='?', const=True, default=True,
                        help="True means fixed length; False means variable length")
    parser.add_argument("--beam_search", type="bool", nargs='?', const=True, default=True,
                        help="if beam search or not. True means beam search; False means not")
    parser.add_argument("--label_smooth", type="bool", nargs='?', const=True, default=True,
                        help="if label smooth or not. True means label smooth; False means not")
    parser.add_argument("--attention_mode", type="bool", nargs='?', const=True, default=True,
                        help="True for hard attention mode; False for soft attention mode")
    parser.add_argument("--input_att_comb_or_not", type="bool", nargs='?', const=True, default=True,
                        help="combine the attention with input or not")
    parser.add_argument("--residual_mode", type="bool", nargs='?', const=True, default=True,
                        help="True for residual mode; False for not residual mode")
    parser.add_argument("--reverse_encoder_input", type="bool", nargs='?', const=True, default=False,
                        help="reverse input or not")
    parser.add_argument("--regularization", type="bool", nargs='?', const=True, default=False,
                        help="regularization or not")


# -----------------------------define the FLAGS parameters for whole model usage.---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------from here, kick off training, evaluation and prediction----------------------------------
def run_main(argv=None):
    # 0. ---------------------------------------(re)set or change  FLAGS------------------------------------------------
    #   reset partial FLAGS
    if FLAGS.max_input_time_steps == 0 and FLAGS.max_output_time_steps == 0:
        FLAGS.__setattr__('max_input_time_steps', src_max_length)
        FLAGS.__setattr__('max_output_time_steps', tgt_max_length)

    FLAGS.__setattr__('infer_max_input_time_steps', FLAGS.max_input_time_steps)
    FLAGS.__setattr__('infer_max_output_time_steps', FLAGS.max_output_time_steps)

    FLAGS.__setattr__('src_bucket_width', math.ceil(FLAGS.max_input_time_steps / FLAGS.num_buckets))
    FLAGS.__setattr__('tgt_bucket_width', math.ceil(FLAGS.max_output_time_steps / FLAGS.num_buckets))

    FLAGS.__setattr__('input_vocab_size', encoder_vocab_size)
    FLAGS.__setattr__('output_vocab_size', decoder_vocab_size)

    print("max_input_time_steps:", FLAGS.max_input_time_steps)
    print("max_output_time_steps:", FLAGS.max_output_time_steps)
    print("input_vocab_size:", FLAGS.input_vocab_size)
    print("output_vocab_size:", FLAGS.output_vocab_size)
    print("src_bucket_width:", FLAGS.src_bucket_width)
    print("tgt_bucket_width:", FLAGS.tgt_bucket_width)

    # ------------------------------------------------------------------------------------------------------------------
    # 1. ----------------------------------------self-defined HPARAM----------------------------------------------------
    params = tf.contrib.training.HParams(
        train_steps=FLAGS.train_steps,
        min_eval_frequency=200,
        min_summary_frequency=200,
        log_steps=200,
        learning_rate=FLAGS.learning_rate,
        warmup_steps=FLAGS.warmup_steps
    )
    # ------------------------------------------------------------------------------------------------------------------
    # 2. ----------------------------define the training and evaluation env parameters----------------------------------
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(
        model_dir=FLAGS.hard_att_model_dir if FLAGS.attention_mode else FLAGS.soft_att_model_dir,
        save_checkpoints_steps=params.min_eval_frequency,
        save_summary_steps=params.min_summary_frequency,
        log_step_count_steps=params.log_steps)
    # ------------------------------------------------------------------------------------------------------------------
    # 3. ------------------------------------------Define Estimator-----------------------------------------------------
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )
    # ------------------------------------------------------------------------------------------------------------------
    # 4. ----------------------------------prepare the training and evluation specs-------------------------------------

    train_input_fn, train_input_hook = get_train_inputs(src=FLAGS.train_src_file_path, tgt=FLAGS.train_tgt_file_path,
                                                        batch_size=FLAGS.batch_size)
    eval_input_fn, eval_input_hook = get_eval_inputs(src=FLAGS.eval_src_file_path, tgt=FLAGS.eval_tgt_file_path,
                                                     batch_size=FLAGS.batch_size)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=params.train_steps, hooks=[train_input_hook])

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, hooks=[eval_input_hook])
    # ------------------------------------------------------------------------------------------------------------------
    # 5. ----------------------------------construct the train_and_evaluate---------------------------------------------
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # ------------------------------------------------------------------------------------------------------------------
    # 6. --------------------------------------------Prediction Process-------------------------------------------------

    pred = []
    pred_out = []
    pred_feature = en_translation_file_processing(FLAGS.infer_src_file_path, FLAGS.infer_max_input_time_steps,
                                                  FLAGS.input_vocab_size)  # [B,T,V]
    pred_label = de_translation_file_processing(FLAGS.infer_tgt_file_path, FLAGS.infer_max_output_time_steps,
                                                FLAGS.output_vocab_size)  # [B,T,V]
    pred_label = np.transpose(pred_label, [1, 0, 2])  # [T,B,V]

    predict_input_fn, pred_input_hook = get_predict_inputs(src=FLAGS.infer_src_file_path,
                                                           batch_size=FLAGS.batch_size)

    predictions = estimator.predict(input_fn=predict_input_fn, hooks=[pred_input_hook])
    j = 0
    for To_i in predictions:
        j += 1
        pred.append(To_i)
        if j % FLAGS.infer_max_output_time_steps == 0:
            pred_out.append(np.reshape(pred, [FLAGS.infer_max_output_time_steps, -1,
                                              FLAGS.output_vocab_size + FLAGS.infer_max_input_time_steps]))
            j = 0
            pred = []

    pred_out = np.concatenate(pred_out, axis=1)
    pred_data = pred_out[:, :, :FLAGS.output_vocab_size]
    pred_image = pred_out[:, :,
                 FLAGS.output_vocab_size:]

    def clean_batch(data):
        out = []
        for each in data:
            if FLAGS.whitespace_or_nonws_slip:
                each_array = each.split()
                each_temp = ''
                for i in each_array:
                    if i != EOS:
                        each_temp += (i + " ")
                    else:
                        break
                each_temp = each_temp.strip()
                out.append(each_temp)
            else:
                each_temp = ''
                for i in each:
                    each_temp += i
                each_temp = each_temp.strip()
                out.append(each_temp)

        return out

    def clean_single(data):
        if FLAGS.whitespace_or_nonws_slip:
            each_array = data.split()
            each_temp = ''
            for i in each_array:
                if i != EOS:
                    each_temp += (i + " ")
                else:
                    break
            out = each_temp.strip()
        else:
            each_temp = ''
            for i in data:
                each_temp += i
            out = each_temp.strip()
        return out


    pred_label_show = clean_batch(de_batches2string(pred_label))
    pred_data_show = clean_batch(de_batches2string(pred_data))

    for label_show, data_show in zip(pred_label_show, pred_data_show):
        print("----------------------------------------------------------------------------------------")
        print(' Expected out String:' + label_show)
        print('Predicted out String:' + data_show)
        print("----------------------------------------------------------------------------------------")

    print("Accuracy:", accuracy(pred_data, pred_label))
    # ---------------------------------------------------Draw-----------------------------------------------------------
    image_out = np.transpose(pred_image, [1, 0, 2])  # [B,To,Ti]

    fig, ax = plt.subplots()
    image_ids = []
    # generate the random show image batch_id list.
    for i in range(len(pred_feature)):
        image_ids.append(random.randint(0, len(pred_feature) - 1))
    for seq_id in image_ids:
        if FLAGS.whitespace_or_nonws_slip:
            xlabel = clean_single(en_batches2string(np.expand_dims(pred_feature[seq_id], axis=1))[0]).split()
            ylabel = clean_single(de_batches2string(np.expand_dims(pred_data[:, seq_id, :], axis=1))[0]).split()
        else:
            xlabel = clean_single(en_batches2string(np.expand_dims(pred_feature[seq_id], axis=1))[0])
            ylabel = clean_single(de_batches2string(np.expand_dims(pred_data[:, seq_id, :], axis=1))[0])
        ax.set_xticks(np.arange(len(xlabel)))
        ax.set_yticks(np.arange(len(ylabel)))
        ax.set_xticklabels(xlabel)
        ax.set_yticklabels(ylabel)
        from matplotlib.font_manager import FontProperties
        myfont = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
        plt.setp(ax.get_yticklabels(), rotation=0, fontproperties=myfont)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", fontproperties=myfont)
        for i in range(len(ylabel)):
            for j in range(len(xlabel)):
                ax.text(j, i, "%.4f" % image_out[seq_id][i, j], ha="center", va="center", color="w",
                        fontproperties=myfont, fontsize=5)

        fig.tight_layout()
        ax.xaxis.tick_top()
        ax.set_xlabel('Input Sequence ----------->')
        ax.set_ylabel('Output Sequence <-----------')
        ax.imshow(image_out[seq_id][:len(ylabel), :len(xlabel)] * 255, cmap='Blues', alpha=1)
        plt.suptitle(str(seq_id) + ' Sentence', fontsize=10)
        plt.pause(2)
    # -----------------------------------------------------Draw---------------------------------------------------------
    # -------------------------------------print out the attention heat map---------------------------------------------


def model_fn(features, labels, mode, params):
    loss = None
    train_op = None
    eval_metric_ops = {}

    def label_smoothing(inputs, l_smooth=True):
        if l_smooth:
            epsilon = 0.05
        else:
            epsilon = 0.0
        K = inputs.get_shape().as_list()[-1]
        return ((1 - epsilon) * inputs) + (epsilon / K)

    if mode != ModeKeys.PREDICT:
        features, feature_len = features
        features = tf.transpose(features)
        labels_in, label_out, label_len = labels
        if FLAGS.fix_or_variable_length:
            tgt_seq_mask = tf.cast(tf.sequence_mask(label_len, FLAGS.max_output_time_steps), tf.float32)  # [B,T]
        else:
            tgt_seq_mask = tf.cast(tf.sequence_mask(label_len), tf.float32)  # [B,T]
        tgt_seq_mask_original = tf.transpose(tgt_seq_mask)  # [T,B]
        tgt_seq_mask = tf.expand_dims(tgt_seq_mask_original, axis=2)  # [T,B,1]
        label_one_hot = tf.one_hot(indices=label_out, depth=FLAGS.output_vocab_size, axis=-1)  # [B, T, V]
        label_one_hot = tf.transpose(label_one_hot, (1, 0, 2))  # [T,B,V]
        label_one_hot_mask = label_one_hot * tgt_seq_mask
        labels_in = tf.transpose(labels_in)  # [T,B]
        logits, softmax = architecture(features, feature_len, labels_in, label_len, mode)
        logits_mask = logits * tgt_seq_mask
        predictions = tf.nn.softmax(logits) * tgt_seq_mask

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label_smoothing(label_one_hot_mask, FLAGS.label_smooth), logits=logits_mask)
        loss = tf.reduce_sum(loss) / tf.count_nonzero(tgt_seq_mask, dtype=tf.float32)
        if FLAGS.regularization:
            regularizer_loss = FLAGS.reg_lambda * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = loss + regularizer_loss

        train_op = get_train_op(loss, params)
        eval_metric_ops = get_eval_metric_ops(tf.argmax(label_one_hot_mask, axis=-1), tf.argmax(predictions, axis=-1),
                                              tgt_seq_mask_original)
    else:
        feature_len = features[:, -1]
        features = features[:, :-1]
        features = tf.transpose(features)  # [B,T]-->[T,B]
        labels_in = None
        label_len = None
        logits, softmax = architecture(features, feature_len, labels_in, label_len, mode)
        if FLAGS.beam_search:
            predictions = tf.concat([logits, softmax], axis=-1)
        else:
            predictions = tf.concat([tf.nn.softmax(logits), softmax], axis=-1)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

def get_train_op(loss, params):
    def get_learning_rate_warmup(hparam):
        warmup_steps = hparam.warmup_steps
        warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
        inv_decay = warmup_factor ** (
            tf.to_float(warmup_steps - tf.train.get_global_step()))

        return tf.cond(
            tf.train.get_global_step() < hparam.warmup_steps,
            lambda: inv_decay * learning_rate,
            lambda: learning_rate,
            name="learning_rate_warump_cond")

    def get_learning_rate_decay(hparam):
        decay_factor = FLAGS.decay_factor
        start_decay_step = int(hparam.train_steps * 4 / 5)
        decay_times = 4
        remain_steps = hparam.train_steps - start_decay_step
        decay_steps = int(remain_steps / decay_times)

        return tf.cond(
            tf.train.get_global_step() < start_decay_step,
            lambda: learning_rate,
            lambda: tf.train.exponential_decay(
                learning_rate,
                (tf.train.get_global_step() - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    learning_rate = tf.constant(params.learning_rate)
    if FLAGS.lr_warmup:
        learning_rate = get_learning_rate_warmup(params)
    if FLAGS.lr_decay:
        learning_rate = get_learning_rate_decay(params)

    tf.summary.scalar("learning_rate", learning_rate)
    trainable_params = tf.trainable_variables()
    opt = None
    if FLAGS.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif FLAGS.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)

    gradients = tf.gradients(loss, trainable_params, colocate_gradients_with_ops=True)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
    train_op = opt.apply_gradients(zip(clipped_gradients, trainable_params), tf.train.get_global_step())

    return train_op
    # ----------------------------------------------------------------
def get_eval_metric_ops(labels, predictions, mask):

    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            weights=mask,
            name='Accuracy')
    }


def architecture(encoder_inputs, encoder_len, decoder_inputs, decoder_len, mode):

    regularizer = tf.keras.regularizers.l2(l=FLAGS.reg_lambda)

    if mode != ModeKeys.PREDICT:
        if FLAGS.fix_or_variable_length:
            src_seq_mask = tf.cast(tf.sequence_mask(encoder_len, FLAGS.max_input_time_steps), tf.float32)  # [B,T]
        else:
            src_seq_mask = tf.cast(tf.sequence_mask(encoder_len), tf.float32)  # [B,T]
    else:
        src_seq_mask = tf.cast(tf.sequence_mask(encoder_len, FLAGS.infer_max_input_time_steps), tf.float32)  # [B,T]
    src_seq_mask = tf.expand_dims(src_seq_mask, axis=2)  # [B,T,1]
    with tf.variable_scope(name_or_scope="shared_parameters", reuse=tf.AUTO_REUSE):
        fw_ht_hs_weights = tf.get_variable(name="fw_ht_hs_weights",
                                           shape=[FLAGS.num_units, FLAGS.num_units],
                                           dtype=tf.float32,
                                           initializer=tf.variance_scaling_initializer(),
                                           regularizer=regularizer)
        bw_ht_hs_weights = tf.get_variable(name="bw_ht_hs_weights",
                                           shape=[FLAGS.num_units, FLAGS.num_units],
                                           dtype=tf.float32,
                                           initializer=tf.variance_scaling_initializer(),
                                           regularizer=regularizer)

        # embedding
        encoding_embedding = tf.get_variable(name='encoding_embedding',
                                             shape=[FLAGS.input_vocab_size, FLAGS.num_units],
                                             dtype=tf.float32,
                                             initializer=tf.variance_scaling_initializer())
        decoding_embedding = tf.get_variable(name='decoding_embedding',
                                             shape=[FLAGS.output_vocab_size, FLAGS.num_units],
                                             dtype=tf.float32,
                                             initializer=tf.variance_scaling_initializer())

        wp_weights = tf.get_variable(name='wp_weights',
                                     shape=[FLAGS.num_units, FLAGS.num_units],
                                     dtype=tf.float32,
                                     initializer=tf.variance_scaling_initializer(),
                                     regularizer=regularizer)

        vp_weights = tf.get_variable(name='vp_weights',
                                     shape=[FLAGS.num_units, 1],
                                     dtype=tf.float32,
                                     initializer=tf.variance_scaling_initializer(),
                                     regularizer=regularizer)

    def projection_layer(inputs):
        with tf.variable_scope(name_or_scope='projection_layer', reuse=tf.AUTO_REUSE):
            outputs = normalize(inputs)
            outputs = tf.layers.dense(inputs=outputs, units=4*FLAGS.num_units, activation=None, name='l1',
                                      kernel_regularizer=regularizer,
                                      activity_regularizer=regularizer
                                      )
            outputs = tf.layers.dense(inputs=outputs, units=FLAGS.output_vocab_size, activation=None, name='l2',
                                      kernel_regularizer=regularizer,
                                      activity_regularizer=regularizer
                                      )
        return outputs

    def normalize(inputs):

        epsilon = 1e-8

        with tf.variable_scope(name_or_scope='ln', reuse=tf.AUTO_REUSE):

            params_shape = inputs.get_shape().as_list()[-1]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

            zeros = lambda: tf.zeros([params_shape], dtype=tf.float32)
            beta = tf.get_variable('beta', initializer=zeros)

            ones = lambda: tf.ones([params_shape], dtype=tf.float32)
            gamma = tf.get_variable('gamma', initializer=ones)
            outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)

        return outputs

    def encoder_lstm_units(input_data, input_length):

        def cell():
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.num_units)
            drop_out = 0.0
            if mode == ModeKeys.TRAIN:
                drop_out = FLAGS.dropout

            if drop_out > 0.0:
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, input_keep_prob=(1.0 - drop_out))
            if FLAGS.residual_mode:
                rnn_cell = tf.nn.rnn_cell.ResidualWrapper(cell=rnn_cell)
            return rnn_cell
        # -------------------------------------------------------------------------------------------------------------------
        if FLAGS.num_layers == 1:
            with tf.variable_scope(name_or_scope='encoding_single_rnn', reuse=tf.AUTO_REUSE):
                cell = cell()
                initial_state = cell.zero_state(tf.shape(input_data)[1], dtype=tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, input_data,
                                                   sequence_length=input_length,
                                                   initial_state=initial_state,
                                                   time_major=True,
                                                   dtype=tf.float32)

        else:

            if FLAGS.bi_or_uni:
                with tf.variable_scope(name_or_scope='encoding_bi_rnn', reuse=tf.AUTO_REUSE):
                    cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(FLAGS.num_layers // 2)])
                    cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(FLAGS.num_layers // 2)])
                    outputs, state = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=input_data,
                        sequence_length=input_length,
                        initial_state_fw=cell_fw.zero_state(tf.shape(input_data)[1], dtype=tf.float32),
                        initial_state_bw=cell_bw.zero_state(tf.shape(input_data)[1], dtype=tf.float32),
                        dtype=tf.float32,
                        time_major=True)

                    _state = []
                    for i in range(2):
                        for j in range(FLAGS.num_layers // 2):
                            _state.append(state[i][j])
                    state = tuple(_state)

            # -----------------------------------------------------------------------------------
            else:
                with tf.variable_scope(name_or_scope='encoding_multi_rnn', reuse=tf.AUTO_REUSE):
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(FLAGS.num_layers)])

                    initial_state = cell.zero_state(tf.shape(input_data)[1], dtype=tf.float32)

                    outputs, state = tf.nn.dynamic_rnn(cell, input_data,
                                                       sequence_length=input_length,
                                                       initial_state=initial_state,
                                                       dtype=tf.float32,
                                                       time_major=True
                                                       )
            # -----------------------------------------------------------------------------------

        return outputs, state

    def decoder_lstm_units(input_data, input_length, init_state):
        def cell():
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.num_units)
            drop_out = 0.0
            if mode == ModeKeys.TRAIN:
                drop_out = FLAGS.dropout

            if drop_out > 0.0:
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, input_keep_prob=(1.0 - drop_out))

            if FLAGS.residual_mode and not FLAGS.input_att_comb_or_not:
                rnn_cell = tf.nn.rnn_cell.ResidualWrapper(cell=rnn_cell)
            return rnn_cell

        if FLAGS.num_layers == 1:
            with tf.variable_scope(name_or_scope='decoding_single_rnn', reuse=tf.AUTO_REUSE):
                cell = cell()
                outputs, state = tf.nn.dynamic_rnn(cell, input_data,
                                                   sequence_length=input_length,
                                                   initial_state=init_state,
                                                   time_major=True,
                                                   dtype=tf.float32
                                                   )

        else:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(FLAGS.num_layers)])
            initial_state = init_state
            with tf.variable_scope(name_or_scope='decoding_multi_rnn', reuse=tf.AUTO_REUSE):
                outputs, state = tf.nn.dynamic_rnn(cell, input_data,
                                                   sequence_length=input_length,
                                                   initial_state=initial_state,
                                                   time_major=True,
                                                   dtype=tf.float32
                                                   )
        return outputs, state

    def soft_or_hard_attention(hidden_states, current_hidden_state):
        if FLAGS.bi_or_uni and FLAGS.num_layers >= 2:
            fw_hidden_states, bw_hidden_states = hidden_states
        else:
            fw_hidden_states, bw_hidden_states = hidden_states, hidden_states
            # ----------------------------------Hard Attention Mode Specific Parts------------------------------------------
        gaussian_distribution = None
        if FLAGS.attention_mode:  # True, means hard attention; False means soft attention

            # local-p  [B,1]
            p_t_1 = tf.matmul(tf.tanh(tf.matmul(current_hidden_state, wp_weights)), vp_weights) # [B,1]
            p_t_2 = tf.sigmoid(p_t_1)
            p_t = tf.to_float(tf.shape(fw_hidden_states)[0]) * p_t_2
            p_t = tf.floor(p_t)

            gaussian_distribution = tf.exp(-2.0 * tf.square(p_t - 0.0) / tf.square(FLAGS.predict_window * 1.0))
            gaussian_distribution = tf.expand_dims(gaussian_distribution, axis=0)
            time = tf.constant(1)
            def time_steps(_time, _gaussion):
                next_gaussion = tf.exp(
                    -2.0 * tf.square(p_t - tf.to_float(_time)) / tf.square(FLAGS.predict_window * 1.0))
                _gaussion = tf.concat([_gaussion, [next_gaussion]], axis=0)  # [time, B, 1]
                _time += 1
                return _time, _gaussion

            _, gaussian_distribution = tf.while_loop(
                cond=lambda time, *_: time < tf.shape(fw_hidden_states)[0],
                body=time_steps,
                loop_vars=[time, gaussian_distribution],
                shape_invariants=[time.get_shape(), tf.TensorShape([None, None, None])]
            )
            gaussian_distribution = tf.transpose(gaussian_distribution, [1, 0, 2])  # [B, T, 1]
        # ----------------------------------Hard Attention Mode Specific Parts------------------------------------------

        # -------------------------------soft attention and hard attention common parts---------------------------------

        fw_hidden_states_copy = tf.transpose(fw_hidden_states, perm=[1, 0, 2]) # [B,T,H]
        bw_hidden_states_copy = tf.transpose(bw_hidden_states, perm=[1, 0, 2]) # [B,T,H]

        fw_coff = tf.matmul(current_hidden_state, fw_ht_hs_weights)  # [B,H]*[H,H]==>[B,H]
        bw_coff = tf.matmul(current_hidden_state, bw_ht_hs_weights)  # [B,H]*[H,H]==>[B,H]

        fw_coff = tf.expand_dims(fw_coff, axis=1) # [B,1,H]
        bw_coff = tf.expand_dims(bw_coff, axis=1) # [B,1,H]

        fw_hidden_states = tf.transpose(fw_hidden_states, perm=[1, 2, 0])  # [B, H, T]
        bw_hidden_states = tf.transpose(bw_hidden_states, perm=[1, 2, 0])  # [B, H, T]

        fw_a_t = tf.transpose(tf.matmul(fw_coff, fw_hidden_states), [0, 2, 1])  # [B, T, 1]
        bw_a_t = tf.transpose(tf.matmul(bw_coff, bw_hidden_states), [0, 2, 1])  # [B, T, 1]
        fw_a_t = tf.exp(fw_a_t) * src_seq_mask
        bw_a_t = tf.exp(bw_a_t) * src_seq_mask

        fw_exp_sum = tf.reduce_sum(fw_a_t, axis=1, keepdims=True)
        bw_exp_sum = tf.reduce_sum(bw_a_t, axis=1, keepdims=True)

        fw_softmax = fw_a_t / fw_exp_sum  # [B, T, 1]
        bw_softmax = bw_a_t / bw_exp_sum  # [B, T, 1]
        # ------------------------------soft attention and hard attention common parts----------------------------------
        # ------------------------------------Hard Attention Mode Specific Parts----------------------------------------
        if FLAGS.attention_mode:  # True, means hard attention; False means soft attention
            fw_softmax = tf.multiply(fw_softmax, gaussian_distribution)  # [B, T, 1]
            bw_softmax = tf.multiply(bw_softmax, gaussian_distribution)  # [B, T, 1]
        # ------------------------------------Hard Attention Mode Specific Parts----------------------------------------
        # ---------------------------------soft attention and hard attention common parts-------------------------------
        fw_context = tf.reduce_sum(tf.multiply(fw_softmax, fw_hidden_states_copy), axis=1)  # output shape: [B, H]
        bw_context = tf.reduce_sum(tf.multiply(bw_softmax, bw_hidden_states_copy), axis=1)  # output shape: [B, H]

        if FLAGS.bi_or_uni and FLAGS.num_layers >=2:
            context = (fw_context + bw_context) / 2.0
            softmax = (fw_softmax + bw_softmax) / 2.0
        else:
            context = fw_context
            softmax = fw_softmax

        current_context_hidden_combined = tf.concat([current_hidden_state, context], axis=-1)

        with tf.variable_scope(name_or_scope="attention_dense", reuse=tf.AUTO_REUSE):
            current_context_hidden_combined = tf.layers.dense(
                inputs=current_context_hidden_combined,
                units=FLAGS.num_units,
                activation=tf.tanh,
                kernel_initializer=tf.orthogonal_initializer,
                kernel_regularizer=regularizer,
                name='al'
            )

        return current_context_hidden_combined, softmax, context
        # --------------------------------soft attention and hard attention common parts--------------------------------

    def encoder(input_data, input_len):
        outputs, state = encoder_lstm_units(input_data=input_data, input_length=input_len)
        return outputs, state

    def next_word_with_union_prob(input_data, pre_union_probability, hidden_states, context, state):

        input_data = tf.nn.embedding_lookup(decoding_embedding, input_data)  # [B, H]
        if FLAGS.input_att_comb_or_not:
            inp_concat = tf.concat([input_data, context], axis=-1)
        else:
            inp_concat = input_data
        inp_concat = tf.expand_dims(inp_concat, 0)
        outputs, state = decoder_lstm_units(input_data=inp_concat, input_length=None, init_state=state)

        current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, outputs[-1])
        if FLAGS.residual_mode and FLAGS.input_att_comb_or_not:
            _output = projection_layer(inputs=current_context_hidden_combined + input_data)
        else:
            _output = projection_layer(inputs=current_context_hidden_combined)
        _output = tf.nn.softmax(_output)
        _output = tf.multiply(_output, pre_union_probability)
        return _output, context, state, softmax

    def beam_search(input_data, hidden_states, context, init_state, softmax):

        group_state = []
        topk_probability_value, topk_probability_pos = tf.nn.top_k(input_data, FLAGS.beam_size)

        top_k_branch = tf.transpose(tf.expand_dims(topk_probability_pos, axis=2), [1, 0, 2])  # [k,B,x]
        total_union_probability = tf.transpose(tf.expand_dims(topk_probability_value, axis=2), [1, 0, 2])  # [k,B,1]
        group_context = tf.tile([context], [FLAGS.beam_size, 1, 1])  # [k,B,H]
        softmax_outputs = tf.tile([softmax], [FLAGS.beam_size, 1, 1, 1])  # [k,B,T,x]
        for i in range(FLAGS.beam_size):
            group_state.append(init_state)

        def getValue(value, pos):
            first_row = pos[0, :] + 1
            first_row = tf.diag(first_row)
            _, batch_num = tf.nn.top_k(first_row, k=1)
            batch_num = tf.tile([batch_num], [FLAGS.beam_size, 1, 1])  # [k,B,1]
            pos = tf.expand_dims(pos, axis=2)  # [k,B,1]
            pos = tf.concat([pos, batch_num], axis=-1)
            value_return = tf.gather_nd(value, pos)

            return value_return

        for i in trange(1, FLAGS.infer_max_output_time_steps):

            probability = []
            softmax = []
            top_k_branch_copy = tf.identity(top_k_branch)  # [k,B,x]
            group_context_copy = []
            group_state_copy = []
            softmax_outputs_copy = tf.identity(softmax_outputs)

            for k in range(FLAGS.beam_size):
                _output, _context_copy, _state_copy, _softmax = next_word_with_union_prob(
                    top_k_branch[k, :, -1], total_union_probability[k], hidden_states,
                    group_context[k], group_state[k])

                group_context_copy.append(_context_copy)  # [.., [B, H]...]
                group_state_copy.append(_state_copy)
                probability.append(_output)  # [[B,V],[B,V],[B,V]...]
                softmax.append(_softmax)  # [[B,T,1],[B,T,1],...]

            probability = tf.stack(probability)  # [K,B,V]
            probability = tf.transpose(probability, [1, 0, 2])  # [B,K,V ]
            probability = tf.reshape(probability, [-1, FLAGS.beam_size * FLAGS.output_vocab_size])  # [B,k*V]
            probability_value, probability_pos = tf.nn.top_k(probability, FLAGS.beam_size)  # [B,k]
            _group = probability_pos // FLAGS.output_vocab_size  # [B,k]
            _group = tf.transpose(_group)  # [K,B]
            _next_BS_input = probability_pos % FLAGS.output_vocab_size  # [B,k]
            _next_BS_input = tf.transpose(_next_BS_input)  # [K,B]

            probability_value = tf.transpose(probability_value)  # [k,B]
            top_k_branch = tf.concat(
                [getValue(top_k_branch_copy, _group), tf.expand_dims(_next_BS_input, axis=2)], axis=-1)  # [K,B,x]

            softmax_outputs = tf.concat([getValue(softmax_outputs_copy, _group),
                                         getValue(softmax, _group)], axis=-1)  # [k,B,Ti,x]

            total_union_probability = tf.expand_dims(probability_value, axis=2)  # [k,B,1]

            group_context = getValue(group_context_copy, _group)  # [k,B,H]

            group_state_copy = tf.stack(group_state_copy)

            if FLAGS.num_layers == 1:
                group_state_copy = tf.transpose(group_state_copy, [0,2,1,3]) # [k,B,2,H]
                group_state = getValue(group_state_copy, _group)  # [k,B,2,H]
            else:

                group_state_copy = tf.transpose(group_state_copy, [0, 3, 1, 2, 4])  # [k,B,x,2,H]
                group_state = getValue(group_state_copy, _group)  # [k,B,x,2,H]

            if FLAGS.num_layers == 1:
                group_state = tf.transpose(group_state, [0,2,1,3]) # [k,2,B,H]
            else:
                group_state = tf.transpose(group_state, [0, 2, 3, 1, 4])  # [k,x,2,B,H]

            group_state_temp = []
            for k in range(FLAGS.beam_size):
                if FLAGS.num_layers == 1:
                    group_state_temp.append(tf.nn.rnn_cell.LSTMStateTuple(c=group_state[k][0], h=group_state[k][1]))
                else:
                    _state_ = []
                    for j in range(FLAGS.num_layers):
                        _state_.append(tf.nn.rnn_cell.LSTMStateTuple(c=group_state[k][j][0], h=group_state[k][j][1]))
                    _state = tuple(_state_)
                    group_state_temp.append(_state)

            group_state = group_state_temp

        output = top_k_branch[0]  # [B,To]
        output = tf.one_hot(indices=output, depth=FLAGS.output_vocab_size, axis=-1)  # [B,To,V]
        output = tf.transpose(output, [1, 0, 2])  # [To, B, V]
        softmax_outputs = tf.transpose(softmax_outputs[0], [2, 0, 1])  # [To, B, Ti]
        return output, softmax_outputs

    def train_eval_decoder(input_data, hidden_states, init_state):

        softmax_outputs = []
        EOS_tag = input_data[0]
        if FLAGS.bi_or_uni and FLAGS.num_layers >= 2:
            current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, hidden_states[1][0])
        else:
            current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, hidden_states[-1])

        if FLAGS.input_att_comb_or_not:
            inp_concat = tf.concat([EOS_tag, tf.zeros_like(context, dtype=tf.float32)], axis=-1)
        else:
            inp_concat = EOS_tag
        inp_concat = tf.expand_dims(inp_concat, 0)  # [1, B, 2*H] or [1, B, H]
        outputs, state = decoder_lstm_units(input_data=inp_concat, input_length=None, init_state=init_state)

        current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, outputs[-1])
        decoder_outputs = tf.expand_dims(current_context_hidden_combined, axis=0)

        state= tf.stack(state)

        time = tf.constant(1)

        def time_steps(_time, _out, _cchc, _context, _state):
            if FLAGS.num_layers == 1:
                _state = tf.nn.rnn_cell.LSTMStateTuple(c=_state[0], h=_state[1])
            else:
                _state_ = []
                for i in range(FLAGS.num_layers):
                    _state_.append(tf.nn.rnn_cell.LSTMStateTuple(c=_state[i][0], h=_state[i][1]))
                _state = tuple(_state_)

            if FLAGS.input_att_comb_or_not:
                inp_concat = tf.concat([input_data[_time], _context], axis=-1)
            else:
                inp_concat = input_data[_time]
            inp_concat = tf.expand_dims(inp_concat, 0)
            _outputs, _state = decoder_lstm_units(input_data=inp_concat, input_length=None, init_state=_state)

            current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, _outputs[-1])

            _out = tf.concat([_out, [current_context_hidden_combined]], axis=0)
            _state = tf.stack(_state)
            _time += 1
            return _time, _out, current_context_hidden_combined, context, _state

        _, decoder_outputs, _, _, _ = tf.while_loop(
            cond=lambda time, *_: time < tf.shape(input_data)[0],
            body=time_steps,
            loop_vars=[time, decoder_outputs, current_context_hidden_combined, context, state],
            shape_invariants=[time.get_shape(), tf.TensorShape([None, None, FLAGS.num_units]),
                              current_context_hidden_combined.get_shape(), context.get_shape(), state.get_shape()]
        )
        if FLAGS.residual_mode and FLAGS.input_att_comb_or_not:
            decoder_outputs += input_data

        return decoder_outputs, softmax_outputs

    def infer_decoder(input_data, hidden_states, init_state):

        decoder_outputs = []
        softmax_outputs = []
        EOS_tag = input_data
        if FLAGS.bi_or_uni and FLAGS.num_layers >= 2:
            current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, hidden_states[1][0])
        else:
            current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, hidden_states[-1])

        if FLAGS.input_att_comb_or_not:
            inp_concat = tf.concat([EOS_tag, tf.zeros_like(context, dtype=tf.float32)], axis=-1)
        else:
            inp_concat = EOS_tag
        inp_concat = tf.expand_dims(inp_concat, 0)  # # [1, B, 2*H] or [1, B, H]
        outputs, state = decoder_lstm_units(input_data=inp_concat, input_length=None, init_state=init_state)

        current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, outputs[-1])

        if FLAGS.beam_search:
            if FLAGS.residual_mode and FLAGS.input_att_comb_or_not:
                _output = projection_layer(inputs=current_context_hidden_combined + input_data)
            else:
                _output = projection_layer(inputs=current_context_hidden_combined)
            _output = tf.nn.softmax(_output)
            decoder_outputs, softmax_outputs = beam_search(_output, hidden_states, context, state, softmax)

        else:
            for i in trange(FLAGS.infer_max_output_time_steps):
                if FLAGS.residual_mode and FLAGS.input_att_comb_or_not:
                    _output = projection_layer(inputs=current_context_hidden_combined + input_data)
                else:
                    _output = projection_layer(inputs=current_context_hidden_combined)
                decoder_outputs.append(_output)
                softmax_outputs.append(softmax)
                if i < (FLAGS.infer_max_output_time_steps - 1):
                    _output = tf.nn.softmax(_output)
                    input_data = tf.nn.embedding_lookup(decoding_embedding, tf.argmax(_output, axis=-1))
                    if FLAGS.input_att_comb_or_not:
                        inp_concat = tf.concat([input_data, context], axis=-1)
                    else:
                        inp_concat = input_data
                    inp_concat = tf.expand_dims(inp_concat, 0)
                    outputs, state = decoder_lstm_units(input_data=inp_concat, input_length=None, init_state=state)
                    current_context_hidden_combined, softmax, context = soft_or_hard_attention(hidden_states, outputs[-1])

        decoder_outputs = tf.stack(decoder_outputs)
        softmax_outputs = tf.reshape(softmax_outputs, [FLAGS.infer_max_output_time_steps, -1,
                                                       FLAGS.infer_max_input_time_steps])

        return decoder_outputs, softmax_outputs

    feature_inputs = tf.nn.embedding_lookup(encoding_embedding, encoder_inputs)  # [T, B, H]

    outputs, state = encoder(input_data=feature_inputs, input_len=encoder_len)

    if mode != ModeKeys.PREDICT:
        label_inputs = tf.nn.embedding_lookup(decoding_embedding, decoder_inputs)  # [T, B, H]

        decoder_outputs, softmax = train_eval_decoder(label_inputs, outputs, state)
        logits = projection_layer(inputs=decoder_outputs)
    else:
        label_EOS = tf.sign(encoder_len)  # EOS_ID
        label_EOS = tf.nn.embedding_lookup(decoding_embedding, label_EOS)  # [B, H]
        logits, softmax = infer_decoder(label_EOS, outputs, state)

    return logits, softmax


# ----------------------------------------------Define data loaders ----------------------------------------------------

class InitializerHook(tf.train.SessionRunHook):

    def __init__(self):
        super(InitializerHook, self).__init__()
        self.initializer_func = None

    def after_create_session(self, session, coord):
        self.initializer_func(session)

def get_train_inputs(src, tgt, batch_size):

    initializer_hook = InitializerHook()

    def train_inputs():
        with tf.name_scope('Training_data'):
            src_vocab_table, tgt_vocab_table = create_vocab_tables(FLAGS.src_vocab_file_path, FLAGS.tgt_vocab_file_path)
            src_datasets = tf.data.TextLineDataset(src)
            tgt_datasets = tf.data.TextLineDataset(tgt)
            src_tgt_datasets = tf.data.Dataset.zip((src_datasets, tgt_datasets))
            if FLAGS.whitespace_or_nonws_slip:
                src_tgt_datasets = src_tgt_datasets.map(
                    lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values))
            else:
                src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (
                    tf.string_split([src], delimiter='').values, tf.string_split([tgt], delimiter='').values))

            src_tgt_datasets = src_tgt_datasets.filter(
                lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

            src_tgt_datasets = src_tgt_datasets.map(
                lambda src, tgt: (src[:FLAGS.max_input_time_steps], tgt[:FLAGS.max_output_time_steps - 1]))

            src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (
                tf.cast(src_vocab_table.lookup(src), tf.int32), tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)))

            if FLAGS.reverse_encoder_input:
                src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (tf.reverse(src, [0]), tgt))

            src_tgt_datasets = src_tgt_datasets.map(
                lambda src, tgt: (src, tf.concat([[EOS_ID], tgt], 0), tf.concat([tgt, [EOS_ID]], 0)))

            src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt_in, tgt_out: (
                src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)))

            def batching_func(x):
                return x.padded_batch(
                    batch_size,
                    padded_shapes=(
                        tf.TensorShape([FLAGS.max_input_time_steps if FLAGS.fix_or_variable_length else None]),
                        # src_input
                        tf.TensorShape([FLAGS.max_output_time_steps if FLAGS.fix_or_variable_length else None]),
                        # tgt_input
                        tf.TensorShape([FLAGS.max_output_time_steps if FLAGS.fix_or_variable_length else None]),
                        # tgt_output
                        tf.TensorShape([]),  # src_len
                        tf.TensorShape([])),  # tgt_len
                    padding_values=(
                        PAD_ID,  # src_input
                        PAD_ID,  # tgt_input
                        PAD_ID,  # tgt_output
                        0,  # src_len -- unused
                        0))  # tgt_len -- unused

            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):

                bucket_id = tf.maximum(src_len // FLAGS.src_bucket_width, tgt_len // FLAGS.tgt_bucket_width)
                return tf.to_int64(tf.minimum(FLAGS.num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)

            batched_dataset = src_tgt_datasets.apply(
                tf.data.experimental.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

            batched_dataset = batched_dataset.shuffle(2000)
            batched_dataset = batched_dataset.repeat(None)

            iterator = batched_dataset.make_initializable_iterator()
            next_feature, next_label_in, next_label_out, feature_len, label_len = iterator.get_next()

            initializer_hook.initializer_func = lambda sess: sess.run(iterator.initializer)
            return (next_feature, feature_len), (next_label_in, next_label_out, label_len)

    return train_inputs, initializer_hook


def get_eval_inputs(src, tgt, batch_size):

    initializer_hook = InitializerHook()

    def eval_inputs():

        with tf.name_scope('Eval_data'):
            src_vocab_table, tgt_vocab_table = create_vocab_tables(FLAGS.src_vocab_file_path, FLAGS.tgt_vocab_file_path)
            src_datasets = tf.data.TextLineDataset(src)
            tgt_datasets = tf.data.TextLineDataset(tgt)
            src_tgt_datasets = tf.data.Dataset.zip((src_datasets, tgt_datasets))
            if FLAGS.whitespace_or_nonws_slip:
                src_tgt_datasets = src_tgt_datasets.map(
                    lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values))
            else:
                src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (
                    tf.string_split([src], delimiter='').values, tf.string_split([tgt], delimiter='').values))

            src_tgt_datasets = src_tgt_datasets.filter(
                lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

            src_tgt_datasets = src_tgt_datasets.map(
                lambda src, tgt: (src[:FLAGS.max_input_time_steps], tgt[:FLAGS.max_output_time_steps - 1]))

            src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (
                tf.cast(src_vocab_table.lookup(src), tf.int32), tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)))

            if FLAGS.reverse_encoder_input:
                src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt: (tf.reverse(src, [0]), tgt))

            src_tgt_datasets = src_tgt_datasets.map(
                lambda src, tgt: (src, tf.concat([[EOS_ID], tgt], 0), tf.concat([tgt, [EOS_ID]], 0)))

            src_tgt_datasets = src_tgt_datasets.map(lambda src, tgt_in, tgt_out: (
                src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)))

            def batching_func(x):
                return x.padded_batch(
                    batch_size,
                    padded_shapes=(
                        tf.TensorShape([FLAGS.max_input_time_steps if FLAGS.fix_or_variable_length else None]),
                        # src_input
                        tf.TensorShape([FLAGS.max_output_time_steps if FLAGS.fix_or_variable_length else None]),
                        # tgt_input
                        tf.TensorShape([FLAGS.max_output_time_steps if FLAGS.fix_or_variable_length else None]),
                        # tgt_output
                        tf.TensorShape([]),  # src_len
                        tf.TensorShape([])),  # tgt_len
                    padding_values=(
                        PAD_ID,  # src_input
                        PAD_ID,  # tgt_input
                        PAD_ID,  # tgt_output
                        0,  # src_len -- unused
                        0))  # tgt_len -- unused

            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):

                bucket_id = tf.maximum(src_len // FLAGS.src_bucket_width, tgt_len // FLAGS.tgt_bucket_width)
                return tf.to_int64(tf.minimum(FLAGS.num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)

            batched_dataset = src_tgt_datasets.apply(
                tf.data.experimental.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

            batched_dataset = batched_dataset.shuffle(2000)
            batched_dataset = batched_dataset.repeat(None)

            iterator = batched_dataset.make_initializable_iterator()
            next_feature, next_label_in, next_label_out, feature_len, label_len = iterator.get_next()

            initializer_hook.initializer_func = lambda sess: sess.run(iterator.initializer)

            return (next_feature, feature_len), (next_label_in, next_label_out, label_len)

    return eval_inputs, initializer_hook


def get_predict_inputs(src, batch_size):

    initializer_hook = InitializerHook()

    def predict_inputs():

        with tf.name_scope('Predict_data'):
            src_vocab_table, tgt_vocab_table = create_vocab_tables(FLAGS.src_vocab_file_path, FLAGS.tgt_vocab_file_path)
            src_datasets = tf.data.TextLineDataset(src)
            if FLAGS.whitespace_or_nonws_slip:
                src_datasets = src_datasets.map(lambda src: tf.string_split([src]).values)
            else:
                src_datasets = src_datasets.map(lambda src: tf.string_split([src], delimiter='').values)

            src_datasets = src_datasets.filter(
                lambda src: tf.size(src) > 0)

            src_datasets = src_datasets.map(
                lambda src: src[:FLAGS.infer_max_input_time_steps])

            src_datasets = src_datasets.map(
                lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

            if FLAGS.reverse_encoder_input:
                src_datasets = src_datasets.map(lambda src: tf.reverse(src, [0]))

            src_datasets = src_datasets.map(lambda src: (src, tf.size(src)))

            src_datasets = src_datasets.padded_batch(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([FLAGS.infer_max_input_time_steps]),  # src_input
                    tf.TensorShape([])),  # src_len
                padding_values=(
                    PAD_ID,  # src_input
                    0))  # src_len

            iterator = src_datasets.make_initializable_iterator()
            next_feature, feature_len = iterator.get_next()

            initializer_hook.initializer_func = lambda sess: sess.run(iterator.initializer)

            return tf.concat([next_feature, tf.expand_dims(feature_len, axis=1)], axis=-1)

    return predict_inputs, initializer_hook


# -----------------------------------------------------Run script ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_Argumets(parser)
    FLAGS, unparsered = parser.parse_known_args()

    src_max_length = max_line_length(FLAGS.train_src_file_path)
    tgt_max_length = max_line_length(FLAGS.train_tgt_file_path)
    encoder_word_int_map, encoder_vocab, encoder_vocab_size = np_vocab_processing(FLAGS.src_vocab_file_path)
    decoder_word_int_map, decoder_vocab, decoder_vocab_size = np_vocab_processing(FLAGS.tgt_vocab_file_path)

    tf.app.run(main=run_main, argv=[sys.argv[0]] + unparsered)