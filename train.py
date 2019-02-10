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
from tacotron.models import SingleSpeakerTacotronV1Model
from hparams import hparams, hparams_debug_string


def train_and_evaluate(hparams, model_dir, source_dir, target_dir):
    interleave_parallelism = get_parallelism(hparams.interleave_cycle_length_cpu_factor,
                                             hparams.interleave_cycle_length_min,
                                             hparams.interleave_cycle_length_max)

    tf.logging.info("Interleave parallelism is %d.", interleave_parallelism)

    def train_input_fn():
        source = (s for s, _ in source_dir)
        target = (t for _, t in target_dir)

        
        tf.data.Dataset.list_files(source_dir)
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
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["blizzard2012", "ljspeech"]
    corpus = importlib.import_module("datasets." + dataset_name)
    corpus_instance = corpus.instantiate(in_dir="", out_dir=data_root)

    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_evaluate(hparams,
                       checkpoint_dir,
                       corpus_instance.training_source_files,
                       corpus_instance.training_target_files,
                       corpus_instance.validation_source_files,
                       corpus_instance.validation_target_files)


if __name__ == '__main__':
    main()
