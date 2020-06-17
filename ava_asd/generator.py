# TensorFlow dataset for the static models.

# Author: Tuan Chien, James Diprose

import os
import pickle
import random
from abc import abstractmethod
from enum import Enum

import cv2
import numpy as np
import tensorflow

from tensorflow.keras.utils import to_categorical
from ava_asd.annotation import load_annotation_file
from ava_asd.config import get_window_sizes
from ava_asd.data_resampler import DataResampler
from ava_asd.utils import get_start_ts, get_frame_filename, get_avr_mode


class DatasetSubset(Enum):
    train = 'train'
    valid = 'valid'
    test = 'test'


class TensorFlowGenerator:

    def __init__(self, indexes: np.ndarray, num_classes: int, batch_size: int, shuffle: bool = True,
                 video_shape=None, audio_shape=None, num_parallel_calls: int = tensorflow.data.experimental.AUTOTUNE,
                 prefetch_buffer_size: int = tensorflow.data.experimental.AUTOTUNE):
        self.indexes = indexes
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_instances = len(indexes)
        self.video_shape = video_shape
        self.audio_shape = audio_shape

        # Create TensorFlow Dataset from instance_id and labels
        dataset = tensorflow.data.Dataset.from_tensor_slices(indexes)

        # Process the image
        dataset = dataset.map(self.tf_load_data, num_parallel_calls=num_parallel_calls)

        # Make batches
        dataset = dataset.batch(self.batch_size)

        # Prefect the batches of epochs
        self.dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    def tf_load_data(self, instance_id):
        [video, audio] = tensorflow.py_function(self.load_inputs, [instance_id],
                                                [tensorflow.float32, tensorflow.float32])
        label = tensorflow.py_function(self.process_label, [instance_id], tensorflow.float32)
        video.set_shape(self.video_shape)
        audio.set_shape(self.audio_shape)
        label.set_shape((self.num_classes,))
        return (audio, video), (label, label, label)

    @abstractmethod
    def process_label(self, instance_id):
        # Covert the label into it's final form, e.g. one hot encode.
        raise NotImplementedError("Please implement TfDatasetGenerator.process_label")

    @abstractmethod
    def load_inputs(self, instance_id: int):
        # Use Ray https://ray.readthedocs.io/en/latest/ to make loading / processing of images truly parallel.
        # TensorFlow Data uses threads, not processes, so the loading is hamstrung by the Python GIL.
        # For image loading / processing, the overhead of receiving images from Ray is lower than not being able
        # to utilize all CPU resources.
        raise NotImplementedError("Please implement TfDatasetGenerator.load_image")


class AvGenerator(TensorFlowGenerator, tensorflow.keras.callbacks.Callback):
    def __init__(self, mode: str, annotations, train_dir, subset: DatasetSubset, batch_size=128, nmfcc=13,
                 video_window_size=20, audio_window_size=80, shuffle=True, height=100, width=100, channels=3, fps=25,
                 dtype='float32', classes={'SPEAKING_AUDIBLE': 0, 'NOT_SPEAKING': 1, 'SPEAKING_NOT_AUDIBLE': 1},
                 augment=True, resample=True, sequence_size=1, small_face_threshold=100, remove_small_faces=False,
                 filter_out=False, filter_classes={}, keep_ratio=1.0, start_ts_filename='start_ts',
                 num_parallel_calls: int = tensorflow.data.experimental.AUTOTUNE,
                 prefetch_buffer_size: int = tensorflow.data.experimental.AUTOTUNE):
        self.audio, self.video = get_avr_mode(mode)
        self.subset = subset
        self.anns = annotations
        self.anns_selected = annotations
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.nmfcc = nmfcc
        self.video_window_size = video_window_size
        self.audio_window_size = audio_window_size
        self.sequence_size = sequence_size
        self.shuffle = shuffle
        self.augment = augment
        self.height = height
        self.width = width
        self.channels = channels
        self.fps = fps
        self.dtype = dtype
        self.classes = classes
        self.num_classes = len(set(classes.values()))
        self.resample = resample
        self.resampler = DataResampler(annotations, classes, small_face_threshold, remove_small_faces, filter_out,
                                       filter_classes, keep_ratio)
        self.keep_ratio = keep_ratio
        self.start_ts_filename = start_ts_filename
        self.resample_data()  # populates anns_selected

        # Video and audio shapes
        video_shape = (video_window_size, height, width, channels)
        audio_shape = (nmfcc, audio_window_size, 1)

        indexes = np.arange(len(self.anns_selected))
        super(AvGenerator, self).__init__(indexes, self.num_classes, batch_size, shuffle=shuffle,
                                          video_shape=video_shape, audio_shape=audio_shape,
                                          num_parallel_calls=num_parallel_calls,
                                          prefetch_buffer_size=prefetch_buffer_size)

    def __str__(self):
        num_speaking = len(self.resampler.speaking)
        num_not_speaking = len(self.resampler.not_speaking)
        total = num_speaking + num_not_speaking

        val = f"AvGenerator.{self.subset.value}:\n" \
              f"\tNum annotations: {len(self.anns)}\n" \
              f"\tNum classes: {self.num_classes}\n" \
              f"\tNum speaking frames: {num_speaking}\n" \
              f"\tNum not-speaking frames: {num_not_speaking}\n" \
              f"\tSpeaking ratio: {num_speaking / total * 100:.1f}%\n" \
              f"\tNot-speaking ratio: {num_not_speaking / total * 100:.1f}%\n"

        if self.subset is DatasetSubset.valid:
            val += f"\tNum annotations validation: {len(self.anns_selected)}\n"
        elif self.resample:
            val += f"\tNum annotations balanced resampling: {len(self.anns_selected)}\n"
        return val

    def targets(self, invert=False):
        targets_ = []
        for anns_window in self.anns_selected:
            ann = anns_window[-1]
            class_id = self.classes[ann.label]
            targets_.append(class_id)

        targets_ = np.array(targets_)
        if invert:
            targets_ = np.logical_not(targets_).astype(float)
        return targets_

    def process_label(self, instance_id: int):
        ann = self.anns_selected[instance_id]

        # Get class_id for annotation based on the last label in the annotation
        last_label = ann[-1].label
        class_id = self.classes[last_label]

        # One hot encode
        class_id_one_hot = to_categorical(class_id, self.num_classes)
        return class_id_one_hot

    def load_inputs(self, instance_id: int):
        anns = self.anns_selected[instance_id]

        video = np.empty(self.video_shape, dtype=self.dtype)
        audio = np.empty(self.audio_shape, dtype=self.dtype)

        self._insert_video(video, anns)
        self._insert_audio(audio, anns)

        return video, audio


    def audio_augmentation(audio):
        '''
        Apply augmentation to audio.
        '''
        pass


    def _insert_audio(self, audio, anns):
        """
        Insert the audio data into the batch.
        """
        audiopath = os.path.join(self.train_dir, anns[0].vid_id, 'mfcc.pkl')
        mfccs = pickle.load(open(audiopath, 'rb'))
        end_idx = int((anns[-1].timestamp - mfccs.timestamp) / mfccs.stride)
        start_idx = int(end_idx - self.audio_window_size)

        for coeff in range(self.nmfcc):
            start = start_idx
            end = start + self.audio_window_size
            audio[coeff] = mfccs.data[coeff][start:end].reshape(self.audio_window_size, 1)

        if self.subset is DatasetSubset.train and self.augment:
            audio_augmentation(audio)


    def video_augmentation(video):
        '''
        Apply augmentation to video.
        '''
        pass


    def _insert_video(self, video, anns):
        """
        Insert the video frames into the batch.
        """
        start_ts = get_start_ts(self.train_dir, anns[0].vid_id, self.start_ts_filename)

        frames = []
        for t, ann in enumerate(anns):
            img = self._get_video_frame(start_ts, ann)
            frames.append(img)

        if self.subset is DatasetSubset.train and self.augment:
            video_augmentation(frames)

        for t in range(len(anns)):
            img = frames[t]
            video[t] = img / 255

    def _get_video_frame(self, start_ts, ann):
        """
        Load a video frame from file and resize to the expected height/width.
        """

        path = get_frame_filename(self.train_dir, ann, start_ts, self.fps)

        if self.channels == 3:
            img = cv2.imread(path)
        elif self.channels == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img_dim = img.shape
        x1 = int(ann.bbox[0] * img_dim[1])
        y1 = int(ann.bbox[1] * img_dim[0])
        x2 = int(ann.bbox[2] * img_dim[1])
        y2 = int(ann.bbox[3] * img_dim[0])

        if self.channels == 3:
            crop = img[y1:y2, x1:x2, :]
        elif self.channels == 1:
            crop = img[y1:y2, x1:x2]

        resized = cv2.resize(crop, (self.height, self.width))

        if self.channels == 1:
            resized = resized.reshape(self.height, self.width, 1)

        return resized

    def resample_data(self):
        # Only use the resampler for the train set when resample is set to True
        if self.subset is DatasetSubset.train and self.resample:
            self.anns_selected = self.resampler.next()
        elif self.keep_ratio < 1.0:
            self.anns_selected = random.sample(self.anns, int(self.keep_ratio * len(self.anns)))
        elif self.shuffle:
            self.anns_selected = random.sample(self.anns, len(self.anns))

    def on_epoch_end(self, epoch, logs=None):
        """
        Maintenance at the start of each epoch, for example shuffling.
        """

        # Only shuffle or resample data at the end of each epoch for the train and valid datasets
        if self.subset is DatasetSubset.train or self.subset is DatasetSubset.valid:
            self.resample_data()

    @staticmethod
    def from_dict(data_path: str, subset: DatasetSubset, config: dict, **kwargs):
        config_ = dict(config)

        # Kwargs is used to override values in the config
        for key, value in kwargs.items():
            config_[key] = value

        # Check compulsory values
        required = ['extracted_path', 'annotation_type', 'mode', 'batch_size', 'nmfcc', 'vid_frame_size',
                    'mfcc_frame_size', 'sequence_size', 'height', 'width', 'channels', 'fps', 'dtype', 'classes',
                    'augment', 'resample', 'small_face_threshold', 'remove_small_faces', 'filter_out',
                    'filter_classes', 'train_keep_ratio', 'valid_keep_ratio', 'test_keep_ratio', 'start_ts']
        for r in required:
            assert r in config_, f"AvGenerator.from_dict: `{r}` is not in `config`."

        # Make paths
        train_dir = os.path.join(data_path, config_['extracted_path'])

        # Load subset specific annotations
        if subset is DatasetSubset.train:
            annotation_type = config_.get('annotation_type')
            annotation_key = f'train_annotations_{annotation_type}'
            annotation_file = os.path.join(data_path, config_[annotation_key])
            annotations = load_annotation_file(annotation_file)
        else:
            annotation_type = config_.get('annotation_type')
            annotation_key = f'test_annotations_{annotation_type}'
            annotation_file = os.path.join(data_path, config_[annotation_key])
            annotations = load_annotation_file(annotation_file)

        # Load subset specific settings
        # Always shuffle train and valid. No need to shuffle test set as it is only used for prediction.
        if subset is DatasetSubset.train:
            shuffle = True
            keep_ratio = config_['train_keep_ratio']
        elif subset is DatasetSubset.valid:
            shuffle = True
            keep_ratio = config_['valid_keep_ratio']
        else:
            shuffle = False
            keep_ratio = config_['test_keep_ratio']

        mode = config_['mode']
        batch_size = config_['batch_size']
        nmfcc = config_['nmfcc']

        vid_frame_size = config_['vid_frame_size']
        mfcc_frame_size = config_['mfcc_frame_size']
        sequence_size = config_['sequence_size']
        video_window_size, audio_window_size = get_window_sizes(mode, vid_frame_size, mfcc_frame_size, sequence_size)

        height = config_['height']
        width = config_['width']
        channels = config_['channels']
        fps = config_['fps']
        dtype = config_['dtype']
        classes = config_['classes']
        augment = config_['augment']
        resample = config_['resample']
        small_face_threshold = config_['small_face_threshold']
        remove_small_faces = config_['remove_small_faces']
        filter_out = config_['filter_out']
        filter_classes = config_['filter_classes']
        start_ts_filename = config_['start_ts']
        num_parallel_calls = config_.get('num_parallel_calls', tensorflow.data.experimental.AUTOTUNE)
        prefetch_buffer_size = config_.get('prefetch_buffer_size', tensorflow.data.experimental.AUTOTUNE)

        gen = AvGenerator(mode, annotations, train_dir, subset, batch_size=batch_size, nmfcc=nmfcc,
                          video_window_size=video_window_size, audio_window_size=audio_window_size, shuffle=shuffle,
                          height=height, width=width, channels=channels, fps=fps, dtype=dtype, classes=classes,
                          augment=augment, resample=resample, keep_ratio=keep_ratio, sequence_size=sequence_size,
                          small_face_threshold=small_face_threshold, remove_small_faces=remove_small_faces,
                          filter_out=filter_out, filter_classes=filter_classes, start_ts_filename=start_ts_filename,
                          num_parallel_calls=num_parallel_calls, prefetch_buffer_size=prefetch_buffer_size)
        return gen
