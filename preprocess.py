# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================


"""
Preprocess dataset
usage: preprocess.py [options] <name> <in_dir> <out_dir>


options:
    --source-only            Process source only.
    --target-only            Process target only.
    -h, --help               Show help message.

"""
from datasets import ljspeech
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Datasets/LJSpeech-1.1'))
    parser.add_argument('--out_dir', default=os.path.expanduser('~/TFRecords/ljspeech'))
    parser.add_argument('--format', default='ljspeech')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    assert args.format in ["blizzard2012", "ljspeech"]


    
    #  if args.format == "ljspeech":
    data = ljspeech.LJSpeech(args.data_dir, args.out_dir)
    target_metadata = data.process_target()
    for i in target_metadata:
            print(i)

    

    #target_num, max_target_len = data.aggregate_target_metadata(target_metadata)
    #print(f"number of target records: {target_num}, max target length: {max_target_len}")


    #     source_meta = instance.process_sources(
    #         instance.text_and_path_rdd(sc))
    #     source_num, max_source_len = instance.aggregate_source_metadata(source_meta)
    #     print(f"number of source records: {source_num}, max source length: {max_source_len}")
