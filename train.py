import tensorflow as tf
import argparse
import os
from random import shuffle
from multiprocessing import cpu_count
from datasets import ljspeech
from tacotron.models import SingleSpeakerTacotronV1Model
from hparams import hparams, hparams_debug_string
from util.tfrecord import parse_preprocessed_target_data

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

def input_fn(load_test=False, load_eval=False, load_traning=False):

  def tfrecord_dataset(filename):
    buffer_size= 3 * 1024 * 1024 # 3 MiB per file
    return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)

  # Build files and setup loading them
  files = tf.data.Dataset.list_files(args.tfrecords_dir)

  dataset = files.apply(tf.data.experimental.parallel_interleave(tfrecord_dataset, cycle_length=4, sloppy=True))
  
  # TODO: This map needs works
  dataset = dataset.map(parse_preprocessed_target_data)
  
  dataset = dataset.shuffle(buffer_size=hparams.suffle_buffer_size)
  dataset = dataset.batch(hparams.batch_size)
  dataset = dataset.prefetch(hparams.prefetch_buffer_size)

  # Build test, eval, and training datasets
  test_size = int(1000)
  eval_size = int(1000)
  
  test_dataset = dataset.take(test_size)
  eval_dataset = dataset.skip(test_size)
  training_dataset = dataset.skip(eval_size+test_size)

  results = dataset
  if load_test:
    results = test_dataset
  if load_eval:
    results = eval_dataset
  if load_traning:
    results = training_dataset
  return results

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--tfrecords_dir', default=os.path.expanduser('~/TFRecords/ljspeech'))
  parser.add_argument('--training_dir', default=os.path.expanduser('~/Training/tacotron2'))
  parser.add_argument('--hparams', default='',
  help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()

  print("Command line args:\n", args)

  hparams.parse(args.hparams)
  #print(hparams_debug_string())

  if not os.path.exists(args.training_dir):
    os.makedirs(args.training_dir)

  checkpoint_dir = os.path.join(args.training_dir, 'checkpoint')
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps,
                                      log_step_count_steps=hparams.log_step_count_steps)

  estimator = SingleSpeakerTacotronV1Model(hparams, checkpoint_dir, config=run_config)

  train_spec = tf.estimator.TrainSpec(input_fn=lambda : input_fn(load_traning=True))
  eval_spec = tf.estimator.EvalSpec(input_fn=lambda : input_fn(load_eval=True),
                                    steps=hparams.num_evaluation_steps,
                                    throttle_secs=hparams.eval_throttle_secs,
                                    start_delay_secs=hparams.eval_start_delay_secs)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # train_and_evaluate(hparams=hparams,
    #                    checkpoint_dir=checkpoint_dir,
    #                    train_dataset=train_dataset,
    #                    eval_dataset=eval_dataset,
    #                    test_dataset=test_dataset)