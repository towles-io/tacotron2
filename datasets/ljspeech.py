# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Preprocess for LJSpeech dataset. """


import numpy as np
import os
import random
from tqdm import tqdm
from util import tfrecord
from util.audio import Audio
from hparams import hparams
from datasets.corpus import Corpus, TargetMetaData, TextAndPath, target_metadata_to_tsv, \
    source_metadata_to_tsv, eos
from functools import reduce


class LJSpeech(Corpus):

    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.audio = Audio(hparams)

    def _extract_text_and_path(self, line, index):
        parts = line.strip().split('|')
        wav_path = os.path.join(self.data_dir, 'wavs', '%s.wav' % parts[0])
        text = parts[2]
        model =  TextAndPath(index, wav_path, None, text)
        return model


    def _extract_all_text_and_path(self):
        index = 1
        with open(os.path.join(self.data_dir, 'metadata.csv'), mode='r', encoding='utf-8') as f:
            for line in f:
                extracted = self._extract_text_and_path(line, index)
                if extracted is not None:
                    yield extracted
                    index += 1

    def _text_to_sequence(self, text):
        text = text.upper() if hparams.convert_to_upper else text
        sequence = [ord(c) for c in text] + [eos]
        sequence = np.array(sequence, dtype=np.int64)
        return sequence

    def process_target(self):
        result = []

        data = self._extract_all_text_and_path()
        counter = 0
        for item in data:
            counter += 1            
         
        for paths in tqdm(self._extract_all_text_and_path(), total=counter, unit='Examples'):
            wav = self.audio.load_wav(paths.wav_path)
            spectrogram = self.audio.spectrogram(wav).astype(np.float32)
            n_frames = spectrogram.shape[1]
            mel_spectrogram = self.audio.melspectrogram(wav).astype(np.float32)
            filename = f"ljspeech-target-{paths.id:05d}.tfrecord"
            filepath = os.path.join(self.out_dir, filename)
            sequence = self._text_to_sequence(paths.text)
            
            tfrecord.write_preprocessed_target_data(paths.id, paths.text, sequence, paths.text, sequence, spectrogram.T, mel_spectrogram.T, filepath)

            result.append(TargetMetaData(paths.id, paths.text, filepath, n_frames))
        return result