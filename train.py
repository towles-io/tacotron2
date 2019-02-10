# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================


"""Trainining script for seq2seq text-to-speech synthesis model.
Usage: train.py [options]

Options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters. [default: ].
    --dataset=<name>             Dataset name.
    --checkpoint=<path>          Restore model from checkpoint path if given.
    -h, --help                   Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import importlib
from random import shuffle
from multiprocessing import cpu_count
from datasets.dataset import DatasetSource
from datasets import ljspeech
from tacotron.models import SingleSpeakerTacotronV1Model
from hparams import hparams, hparams_debug_string
import os


def train_and_evaluate(hparams, checkpoint_dir, 
                       train_dataset,
                       eval_dataset,
                       test_dataset)

    interleave_parallelism = get_parallelism(hparams.interleave_cycle_length_cpu_factor,
                                             hparams.interleave_cycle_length_min,
                                             hparams.interleave_cycle_length_max)

    tf.logging.info("Interleave parallelism is %d.", interleave_parallelism)
    def train_input_fn():


        tf.data.Data
        dataset = DatasetSource.create_from_tfrecord_files(source, target, hparams,
                                                           cycle_length=interleave_parallelism,
                                                           buffer_output_elements=hparams.interleave_buffer_output_elements,
                                                           prefetch_input_elements=hparams.interleave_prefetch_input_elements)
        dataset = dataset.prepare_and_zip()
        dataset = dataset.filter_by_max_output_length()
        dataset = dataset.shuffle_and_repeat(hparams.suffle_buffer_size)
        dataset = dataset.group_by_batch()
        dataset = dataset.prefetch(hparams.prefetch_buffer_size)
        return dataset

    def eval_input_fn():
        source_and_target_files = list(zip(eval_source_files, eval_target_files))
        shuffle(source_and_target_files)
        source = tf.data.TFRecordDataset([s for s, _ in source_and_target_files])
        target = tf.data.TFRecordDataset([t for _, t in source_and_target_files])

        dataset = DatasetSource(source, target, hparams)
        dataset = dataset.prepare_and_zip().filter_by_max_output_length().repeat().group_by_batch(batch_size=1)
        return dataset.dataset

    run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps,
                                        log_step_count_steps=hparams.log_step_count_steps)
    estimator = SingleSpeakerTacotronV1Model(hparams, model_dir, config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=hparams.num_evaluation_steps,
                                      throttle_secs=hparams.eval_throttle_secs,
                                      start_delay_secs=hparams.eval_start_delay_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def get_parallelism(factor, min_value, max_value):
    return min(max(int(cpu_count() * factor), min_value), max_value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords_dir', default=os.path.expanduser('~/TFRecords/ljspeech'))
    parser.add_argument('--training_dir', default=os.path.expanduser('~/Training/tacotron2'))
    parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()



    print("Command line args:\n", args)

    args.training_dir = os.path.expanduser(args.training_dir)
    args.tfrecords_dir = os.path.expanduser(args.tfrecords_dir)

    if not os.path.exists(args.training_dir):
        os.makedirs(args.training_dir)

    if not os.path.exists(args.tfrecords_dir):
        os.makedirs(args.tfrecords_dir)

    checkpoint_dir = os.path.join(args.training_dir, 'checkpoint')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    hparams.parse(args.hparams)
    print(hparams_debug_string())

    tf.logging.set_verbosity(tf.logging.INFO)
    #dataset_list = ljspeech.LJSpeech(data_dir="", out_dir=args.tfrecords_dir)


    filenames = tf.data.Dataset.list_files(args.tfrecords_dir)
    full_dataset = tf.data.TFRecordDataset(filenames)

    #dataset = dataset.map(parse_preprocessed_source_data)

    dataset_size = len(filenames)
    train_size = int(0.7 * dataset_size)
    eval_size = int(0.15 * dataset_size)
    test_size = int(0.15 * dataset_size)

    full_dataset = full_dataset.shuffle()
    full_dataset = full_dataset.batch(hparams.batch_size)
    full_dataset = full_dataset.prefetch(buffer_size=1)


    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    eval_dataset = test_dataset.skip(eval_size)
    test_dataset = test_dataset.take(test_size) # set to take whats left. 



    train_and_evaluate(hparams=hparams,
                       checkpoint_dir=checkpoint_dir,
                       train_dataset=train_dataset,
                       eval_dataset=eval_dataset,
                       test_dataset=test_dataset)


if __name__ == '__main__':
    main()
