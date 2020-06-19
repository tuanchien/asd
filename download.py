# Copyright 2020 Tuan Chien, James Diprose
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download script for the AVA active speaker dataset.
# It relies on the information provided by
# - https://github.com/cvdfoundation/ava-dataset
# - https://research.google.com/ava/download.html

# Author: Tuan Chien, James Diprose

import csv
import os
from typing import Union

import click
from tensorflow.keras.utils import get_file

from ava_asd.config import read_config
from ava_asd.utils import create_dir

# URLs to retrieve annotations and videos
ava_asd_train_ann_url = 'https://research.google.com/ava/download/ava_activespeaker_train_v1.0.tar.bz2'
ava_asd_test_ann_url = 'https://research.google.com/ava/download/ava_activespeaker_val_v1.0.tar.bz2'
ava_asd_train_ann_hash = '0e6ebaacfb3a554199e319e0c26a1ec0'
ava_asd_test_ann_hash = 'c00210f1d62caaf32759afb92b4562cc'
asd_vid_url_prefix = 'https://s3.amazonaws.com/ava-dataset/trainval/'

# Video URLs
ava_asd_vids_url = 'https://raw.githubusercontent.com/tuanchien/asd/master/ava_speech_file_names_v1.csv'
ava_asd_vids_url_hash = 'aa990d3a628b7cb0c181572e600d79c0'


@click.group()
def main():
    pass


def download_file(url: str, file_name: str, save_path: str, file_hash: Union[str, None], extract=False) -> str:
    """
    Download and extract a file from the url and save it to the destination path. Checks if video has already been
    downloaded with the file_hash and skips downloading it if it has already been downloaded.
    """

    return get_file(file_name, url, cache_subdir='', file_hash=file_hash, hash_algorithm='md5', cache_dir=save_path,
                    extract=extract)


def get_vid_urls(download_path):
    """
    Process the list of urls, and return a dictionary of {vid_id:url}.
    """
    print('Fetching url list')
    path = download_file(ava_asd_vids_url, 'urls.txt', download_path, ava_asd_vids_url_hash)
    vids = {}

    with open(path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['file_name']
            file_hash = row['md5']
            if file_hash == '':
                file_hash = None

            vid_id = file_name.strip().split('.')[0]
            vids[vid_id] = {'url': f'{asd_vid_url_prefix}{file_name}', 'file_name': file_name, 'file_hash': file_hash}

    return vids


def get_annotated_vids(train_ann_dir, test_ann_dir):
    """
    Get a list of video ids.
    """
    train = os.listdir(train_ann_dir)
    test = os.listdir(test_ann_dir)
    trunc_size = len('-activespeaker.csv')
    vids = []

    for f in train:
        vids.append(f[:-trunc_size])
    for f in test:
        vids.append(f[:-trunc_size])

    return vids


@main.command()
@click.argument('config-file', type=click.File('r'))
@click.argument('data-path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def annotations(config_file, data_path):
    """ Download and extract the annotations.

    CONFIG_FILE: the config file with settings for the experiment.
    DATA_PATH: the path to the root folder where the annotations will be saved.
    """

    # Read configuration
    config = read_config(config_file.name)
    download_path = os.path.join(data_path, config['download_path'])
    create_dir(download_path)

    print('Fetching training annotations')
    download_file(ava_asd_train_ann_url, 'train.tar.bz2', download_path, ava_asd_train_ann_hash, extract=True)

    print('Fetching testing annotations')
    download_file(ava_asd_test_ann_url, 'test.tar.bz2', download_path, ava_asd_test_ann_hash, extract=True)


@main.command()
@click.argument('config-file', type=click.File('r'))
@click.argument('data-path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def videos(config_file, data_path):
    """ Download the videos used in the annotations.

    CONFIG_FILE: the config file with settings for the experiment.
    DATA_PATH: the path to the root folder where the videos will be saved.
    """

    # Read configuration
    config = read_config(config_file.name)

    vid_save_path = os.path.join(data_path, config['vid_save_path'])
    download_path = os.path.join(data_path, config['download_path'])
    train_ann_dir = os.path.join(data_path, config['train_ann_dir'])
    test_ann_dir = os.path.join(data_path, config['test_ann_dir'])

    create_dir(vid_save_path)
    create_dir(download_path)

    vid_urls = get_vid_urls(download_path)
    annotated_vids = get_annotated_vids(train_ann_dir, test_ann_dir)

    for vid_id in annotated_vids:
        if vid_id in vid_urls:
            vid = vid_urls[vid_id]
            url = vid['url']
            file_name = vid['file_name']
            file_hash = vid['file_hash']

            print(f'Downloading: {url}')
            download_file(url, file_name, vid_save_path, file_hash)
        else:
            print(f"Warning: video url list does not contain annotated video with id: {vid_id}. Skipping.")


if __name__ == "__main__":
    main()
