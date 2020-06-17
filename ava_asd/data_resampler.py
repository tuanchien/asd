# Helper class to resample annotations for training.
# Author: Tuan Chien, James Diprose

import random


class DataResampler:
    """
    Helper class to resample the dataset after each epoch.
    """

    def __init__(self, annotations, class_ids, small_face_threshold, remove_small_faces, filter_out, filter_classes,
                 train_keep_ratio):
        self.class_ids = class_ids
        self.face_threshold_size = small_face_threshold
        self.remove_small_faces = remove_small_faces
        self.filter_out = filter_out
        self.filter_classes = filter_classes
        self.keep_ratio = train_keep_ratio

        self.annotations = self.filter_annotations(annotations)
        self.speaking, self.not_speaking = self.speaking_partition()

    def is_speaking(self, sequence):
        """
        Determine whether a sequence is speaking. The criteria for now is to see if the last label in the sequence is a
        speaking label. Another criteria could be to see if the entire sequence is speaking or something in between.
        """
        label = sequence[-1].label
        return self.class_ids[label]

    def has_small_face(self, sequence):
        """
        Determine whether the sequence has a face that's too small.
        """
        for ann in sequence:
            width, height = ann.face_size
            if min(width, height) < self.face_threshold_size:
                return True
        return False

    def has_label(self, sequence, labels):
        """
        Check if the sequence has the labels.
        """
        for element in sequence:
            if element in labels:
                return True
        return False

    def filter_annotations(self, annotations):
        """
        Apply filter to remove certain kinds of annotations.
        """
        filtered = []
        for sequence in annotations:
            if self.remove_small_faces and self.has_small_face(sequence):
                continue
            if self.filter_out and self.has_label(sequence, self.filter_classes):
                continue
            filtered.append(sequence)
        return filtered

    def __len__(self):
        return self.min_length() * 2

    def speaking_partition(self):
        """
        Get index lists to the speaking and non speaking indices of the annotations.
        """
        speak = []
        not_speak = []
        for i, sequence in enumerate(self.annotations):
            if self.is_speaking(sequence):
                speak.append(i)
            else:
                not_speak.append(i)
        return speak, not_speak

    def min_length(self):
        speaking_len = len(self.speaking)
        not_speaking_len = len(self.not_speaking)
        return int(min(speaking_len, not_speaking_len) * self.keep_ratio)

    def resample(self):
        """
        Resample data to get a balanced dataset.
        """
        n = self.min_length()
        rspeak_indices = random.sample(self.speaking, n)
        rnspeak_indices = random.sample(self.not_speaking, n)
        rspeak = []
        rnspeak = []
        for i in rspeak_indices:
            rspeak.append(self.annotations[i])
        for i in rnspeak_indices:
            rnspeak.append(self.annotations[i])
        return rspeak, rnspeak

    def next(self):
        """
        Get the next training and validation set. Validation set will be the same for the same for now.
        """
        speaking, nspeaking = self.resample()
        merged = speaking + nspeaking
        random.shuffle(merged)
        return merged
