# Author: Tuan Chien, James Diprose

import datetime
import glob
import os
from collections import OrderedDict
from timeit import default_timer as timer

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from click_option_group import optgroup
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup
from natsort import natsorted
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from tensorflow.keras.metrics import AUC

from ava_asd.activitynet_evaluate import merge_groundtruth_and_predictions, calculate_precision_recall, \
    compute_average_precision, make_uids
from ava_asd.config import get_optimiser, get_model, get_loss_weights, read_config
from ava_asd.generator import AvGenerator, DatasetSubset
from ava_asd.utils import set_gpu_memory_growth, save_csv, WeightsType
from ava_asd.vis import plot_confusion_matrix


def print_results(weights, loss, audio_acc, video_acc, av_acc, auroc, aupr):
    print('===============================================================================')
    print(f'Evaluating weights file: {weights}')
    print(f'Total loss: {loss:.4f}')
    print(f'Audio accuracy: {audio_acc:.4f}')
    print(f'Video accuracy: {video_acc:.4f}')
    print(f'AV accuracy: {av_acc:.4f}')
    print(f'Area under ROC (auROC): {auroc:.4f}')
    print(f'Area under Precision-Recall: {aupr:.4f}')
    print('===============================================================================')


def evaluate_legacy(model, weights, test_gen, loss, optimiser, loss_weights):
    # Get loss weights
    metrics = ['accuracy', AUC(curve='ROC', name='auroc'), AUC(curve='PR', name='aupr')]

    # Compile model
    model.compile(loss=loss, optimizer=optimiser, metrics=metrics, loss_weights=loss_weights)

    # Evaluate
    print(f'Evaluating weights file: {weights}')
    output = model.evaluate(test_gen.dataset)

    # Report results
    loss = output[0]
    audio_acc = output[4]
    video_acc = output[7]
    av_acc = output[10]
    auroc = output[11]
    aupr = output[12]
    print_results(weights, loss, audio_acc, video_acc, av_acc, auroc, aupr)


def calc_map_activity_net(annotations, y_pred_scores):
    ground_truth, predictions = make_groundtruth_and_predictions(annotations, y_pred_scores)
    merged = merge_groundtruth_and_predictions(ground_truth, predictions)
    precision, recall = calculate_precision_recall(merged)
    return compute_average_precision(precision, recall)


def make_groundtruth_and_predictions(annotations, y_pred_scores):
    ground_truth = []
    predictions = []

    for ann_window, score in zip(annotations, y_pred_scores):
        # Get last annotation in window, which is what we are predicting for
        ann = ann_window[-1]

        # Make ground truth row
        x1, y1, x2, y2 = ann.bbox
        g_row = [ann.vid_id, ann.timestamp, x1, y1, x2, y2, ann.label, ann.face_id]

        # Make prediction row
        p_row = [ann.vid_id, ann.timestamp, x1, y1, x2, y2, 'SPEAKING_AUDIBLE', ann.face_id, score]

        # Append rows to lists
        ground_truth.append(g_row)
        predictions.append(p_row)

    # Make ground truth and prediction data frames
    ground_truth_cols = ['video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1', 'entity_box_x2',
                         'entity_box_y2', 'label', 'entity_id']
    prediction_cols = ground_truth_cols + ['score']
    df_ground_truth = pd.DataFrame(data=ground_truth, columns=ground_truth_cols)
    df_predictions = pd.DataFrame(data=predictions, columns=prediction_cols)

    make_uids(df_ground_truth)
    make_uids(df_predictions)

    return df_ground_truth, df_predictions


def load_orig_annotations(ann_dir):
    annotations = pd.DataFrame()
    files = os.listdir(ann_dir)
    for f in files:
        ann_file = os.path.join(ann_dir, f)
        df = pd.read_csv(ann_file, header=None)
        annotations = annotations.append(df)
    return annotations


def create_annotation_lookup(annotations, scores):
    lookup = {}
    for ann_window, score in zip(annotations, scores):
        ann = ann_window[-1]
        row = [ann.vid_id, ann.timestamp, ann.face_id, score]
        if ann.vid_id not in lookup:
            lookup[ann.vid_id] = {}
        if ann.face_id not in lookup[ann.vid_id]:
            lookup[ann.vid_id][ann.face_id] = []
        lookup[ann.vid_id][ann.face_id].append(row)

    for vidkey,vidfaces in lookup.items():
        for facekey,faceval in vidfaces.items():
            lookup[vidkey][facekey] = sorted(faceval, key=lambda x: x[1])

    return lookup


def find_closest_score(annotations, timestamp):
    '''
    Do a binary search for closest timestamp, and use the score for that.
    '''
    tidx = 1 # timestamp index
    sidx = 3 # score index
    n = len(annotations)
    k = 0
    b = int(n/2)

    while b >= 1:
        while k+b < n and annotations[k+b][tidx] <= timestamp:
            k += b
        b = int(b/2)

    nidx = np.clip(k+1, 0, n-1)
    tsdelta_k = abs(timestamp - annotations[k][tidx])
    tsdelta_n = abs(timestamp - annotations[nidx][tidx])

    if tsdelta_k <= tsdelta_n:
        return annotations[k][sidx]
    return annotations[nidx][sidx]
        

def calc_map_activity_net_orig(lookup, annotations):
    '''
    Calculate mAP against original annotations. Try to match scores with closest timestamp to original annotations.
    '''
    ground_truth = annotations.copy()
    ground_truth.columns = ['video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1', 'entity_box_x2',
                         'entity_box_y2', 'label', 'entity_id']
    n = len(ground_truth)
    predictions = ground_truth.copy()
    predictions['score'] = 0.0
    predictions.iloc[:, predictions.columns.get_loc('label')] = 'SPEAKING_AUDIBLE'

    for i in range(n):
        vidid = predictions.iat[i, predictions.columns.get_loc('video_id')]
        entityid = predictions.iat[i, predictions.columns.get_loc('entity_id')]
        timestamp = predictions.iat[i, predictions.columns.get_loc('frame_timestamp')]

        if entityid not in lookup[vidid]:
            continue
        score = find_closest_score(lookup[vidid][entityid], timestamp)
        predictions.iat[i, predictions.columns.get_loc('score')] = score

    make_uids(ground_truth)
    make_uids(predictions)
    merged = merge_groundtruth_and_predictions(ground_truth, predictions)
    precision, recall = calculate_precision_recall(merged)
    return compute_average_precision(precision, recall)


def get_scores(y_pred, axis=0):
    scores = []
    for y in y_pred:
        scores.append(y[axis])
    return np.array(scores)


def evaluate(model, weights, test_gen, test_ann_dir=None, not_speaking_label=0, speaking_label=1) -> OrderedDict:
    result = OrderedDict()

    # Evaluate
    print(f'Predicting for weights file: {weights}')
    y_audio_pred, y_video_pred, y_main_pred = model.predict(test_gen.dataset, verbose=1)

    # Make sure that the length of the result matches the annotations
    assert len(test_gen.anns_selected) == len(y_main_pred), f"len(test_gen.anns_selected) != len(y_main_pred)"

    # Get ground truth
    # Invert labels so that speaking is 1 and not-speaking is 0
    y_true = test_gen.targets(invert=True)
    result['y_true'] = y_true

    #####################################################
    # Make classification report and confusion matrices
    #####################################################

    # Use argmin to return integer class_id values
    #
    # The model defines: speaking as 0 and not-speaking as 1
    # The ground truth defines: speaking as 1 and not-speaking as 0
    #
    # Model Speaking Example:
    # Using argmin will choose the index 1, which means speaking in the ground truth
    #   0    1   (Indexes)
    # [0.9 0.1]  (Values)
    #
    # Model Non-Speaking Example
    # Using argmin will choose the index 0, which means non-speaking in the ground truth
    #   0    1   (Indexes)
    # [0.1 0.9]  (Values)

    y_audio_class_ids = np.argmin(y_audio_pred, axis=1)
    y_video_class_ids = np.argmin(y_video_pred, axis=1)
    y_main_class_ids = np.argmin(y_main_pred, axis=1)

    result['y_audio_class_ids'] = y_audio_class_ids
    result['y_video_class_ids'] = y_video_class_ids
    result['y_main_class_ids'] = y_main_class_ids

    #####################################
    # Calculate Accuracy
    #####################################

    result['audio_accuracy'] = accuracy_score(y_true, y_audio_class_ids)
    result['video_accuracy'] = accuracy_score(y_true, y_video_class_ids)
    result['main_accuracy'] = accuracy_score(y_true, y_main_class_ids)

    #####################################
    # Calculate ActivityNet mAP results
    #####################################

    # Change to axis=1 if speaking and non-speaking labels are switched in config.yaml
    y_audio_scores = get_scores(y_audio_pred, axis=0)
    y_video_scores = get_scores(y_video_pred, axis=0)
    y_main_scores = get_scores(y_main_pred, axis=0)

    annotations = test_gen.anns_selected
    y_audio_map = calc_map_activity_net(annotations, y_audio_scores)
    y_video_map = calc_map_activity_net(annotations, y_video_scores)
    y_main_map = calc_map_activity_net(annotations, y_main_scores)

    result['audio_map'] = y_audio_map
    result['video_map'] = y_video_map
    result['main_map'] = y_main_map

    #####################################
    # Calculate ActivityNet mAP results
    # on original annotations
    #####################################

    audio_lookup = create_annotation_lookup(annotations, y_audio_scores)
    video_lookup = create_annotation_lookup(annotations, y_video_scores)
    main_lookup = create_annotation_lookup(annotations, y_main_scores)

    # Load original test annotations
    orig_annotations = load_orig_annotations(test_ann_dir)
    y_audio_omap = calc_map_activity_net_orig(audio_lookup, orig_annotations)
    y_video_omap = calc_map_activity_net_orig(video_lookup, orig_annotations)
    y_main_omap = calc_map_activity_net_orig(main_lookup, orig_annotations)

    result['orig_audio_map'] = y_audio_omap
    result['orig_video_map'] = y_video_omap
    result['orig_main_map'] = y_main_omap

    #####################################
    # Calculate scikit-learn AP
    #####################################

    y_audio_ap_sp = average_precision_score(y_true, y_audio_scores, pos_label=speaking_label)
    y_video_ap_sp = average_precision_score(y_true, y_video_scores, pos_label=speaking_label)
    y_main_ap_sp = average_precision_score(y_true, y_main_scores, pos_label=speaking_label)

    result['audio_ap_sp'] = y_audio_ap_sp
    result['video_ap_sp'] = y_video_ap_sp
    result['main_ap_sp'] = y_main_ap_sp

    y_audio_ap_ns = average_precision_score(y_true, y_audio_scores, pos_label=not_speaking_label)
    y_video_ap_ns = average_precision_score(y_true, y_video_scores, pos_label=not_speaking_label)
    y_main_ap_ns = average_precision_score(y_true, y_main_scores, pos_label=not_speaking_label)

    result['audio_ap_ns'] = y_audio_ap_ns
    result['video_ap_ns'] = y_video_ap_ns
    result['main_ap_ns'] = y_main_ap_ns

    #####################################
    # Calculate scikit-learn AUC
    #####################################
    y_audio_auc = roc_auc_score(y_true, y_audio_scores)
    y_video_auc = roc_auc_score(y_true, y_video_scores)
    y_main_auc = roc_auc_score(y_true, y_main_scores)

    result['audio_auc'] = y_audio_auc
    result['video_auc'] = y_video_auc
    result['main_auc'] = y_main_auc

    return result


def display_evaluation(result: OrderedDict):
    # Parameters
    classes = ['Not-speaking', 'Speaking']

    #####################################################
    # Display classification report and confusion matrices
    #####################################################

    y_true = result['y_true']
    y_audio_class_ids = result['y_audio_class_ids']
    y_video_class_ids = result['y_video_class_ids']
    y_main_class_ids = result['y_main_class_ids']

    # Print classification reports
    report = classification_report(y_true, y_audio_class_ids)
    print("Classification Report: Audio")
    print(report)

    report = classification_report(y_true, y_video_class_ids)
    print("Classification Report: Video")
    print(report)

    report = classification_report(y_true, y_main_class_ids)
    print("Classification Report: Audio & Video")
    print(report)

    # Print confusion matrices
    normalize = True
    plot_confusion_matrix(y_true, y_audio_class_ids, classes=classes, normalize=normalize,
                          title='Confusion Matrix: Audio')
    plt.show()

    plot_confusion_matrix(y_true, y_video_class_ids, classes=classes, normalize=normalize,
                          title='Confusion Matrix: Video')
    plt.show()

    plot_confusion_matrix(y_true, y_main_class_ids, classes=classes, normalize=normalize,
                          title='Confusion Matrix: Audio & Video')
    plt.show()

    #####################################
    # Display ActivityNet mAP results
    #####################################

    y_audio_map = result['audio_map']
    y_video_map = result['video_map']
    y_main_map = result['main_map']

    y_orig_audio_map = result['orig_audio_map']
    y_orig_video_map = result['orig_video_map']
    y_orig_main_map = result['orig_main_map']

    print("ActivityNet mAP")
    print(f"\tAudio: {y_audio_map:.5f}")
    print(f"\tVideo: {y_video_map:.5f}")
    print(f"\tAudio & Video: {y_main_map:.5f}")

    print("ActivityNet mAP (orig)")
    print(f"\tAudio: {y_orig_audio_map:.5f}")
    print(f"\tVideo: {y_orig_video_map:.5f}")
    print(f"\tAudio & Video: {y_orig_main_map:.5f}")

    #####################################
    # Display scikit-learn AP
    #####################################

    y_audio_ap_sp = result['audio_ap_sp']
    y_video_ap_sp = result['video_ap_sp']
    y_main_ap_sp = result['main_ap_sp']

    print("\nScikit-learn AP Speaking")
    print(f"\tAudio: {y_audio_ap_sp:.5f}")
    print(f"\tVideo: {y_video_ap_sp:.5f}")
    print(f"\tAudio & Video: {y_main_ap_sp:.5f}")

    y_audio_ap_ns = result['audio_ap_ns']
    y_video_ap_ns = result['video_ap_ns']
    y_main_ap_ns = result['main_ap_ns']

    print("\nScikit-learn AP Not-speaking")
    print(f"\tAudio: {y_audio_ap_ns:.5f}")
    print(f"\tVideo: {y_video_ap_ns:.5f}")
    print(f"\tAudio & Video: {y_main_ap_ns:.5f}")

    #####################################
    # Display scikit-learn AUC
    #####################################

    y_audio_auc = result['audio_auc']
    y_video_auc = result['video_auc']
    y_main_auc = result['main_auc']

    print("\nScikit-learn AUC")
    print(f"\tAudio AUC: {y_audio_auc:.5f}")
    print(f"\tVideo AUC: {y_video_auc:.5f}")
    print(f"\tAudio & Video AUC: {y_main_auc:.5f}\n")


@click.command()
@click.argument('config-file', type=click.File('r'))
@click.argument('data-path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@optgroup.group('Weights', help='A weights file or folder to evaluate.', cls=RequiredMutuallyExclusiveOptionGroup)
@optgroup.option('--weights-file', type=WeightsType(), help='A path to the weights .hdf5 file to load into the model.')
@optgroup.option('--weights-path', type=click.Path(exists=True, file_okay=False, dir_okay=True),
                 help='A path to a directory containing multiple weights files. All weights files in the directory'
                      'will be evaluated and the results printed to std out.')
@click.option('--legacy', is_flag=True, help='Run in legacy evaluation mode. Only applicable when using the '
                                             '--weights-file argument.')
def main(config_file, data_path, weights_file, weights_path, legacy):
    """ Evaluate a model based on the test set.

    CONFIG_FILE: the config file with settings for the experiment.
    DATA_PATH: the path to the folder with the data files.
    WEIGHTS: the weights to load into the model.
    """

    # Start time for measuring experiment
    start = timer()

    if weights_path is not None and legacy:
        print("Error: --legacy can only be used with --weights-file, not --weights-path")
    else:
        # Enable memory growth on GPU
        set_gpu_memory_growth(True)

        # Read config
        config = read_config(config_file.name)

        # Get test annotations directory
        test_ann_dir = os.path.join(data_path, config['test_ann_dir'])

        # Get loss weights
        optimiser = get_optimiser(config)
        loss_weights = get_loss_weights(config)

        if weights_file is not None:
            # Load model
            model, loss = get_model(config, weights_file=weights_file)

            # Compile model
            model.compile(loss=loss, optimizer=optimiser, metrics=['accuracy'], loss_weights=loss_weights)

            # Data generator
            test_gen = AvGenerator.from_dict(data_path, DatasetSubset.test, config)

            if not legacy:
                result = evaluate(model, weights_file, test_gen, test_ann_dir)
                display_evaluation(result)
            else:
                evaluate_legacy(model, weights_file, test_gen, loss, optimiser, loss_weights)
        elif weights_path is not None:
            # Load model
            model, loss = get_model(config, weights_file=None)

            # Compile model
            model.compile(loss=loss, optimizer=optimiser, metrics=['accuracy'], loss_weights=loss_weights)

            # List all weights in directory
            weights_files = glob.glob(f"{weights_path}/*.hdf5")
            weights_files = natsorted(weights_files)

            # Data generator
            test_gen = AvGenerator.from_dict(data_path, DatasetSubset.test, config)

            # Evaluate each weights file
            columns = ['weights', 'audio_accuracy', 'video_accuracy', 'main_accuracy', 'audio_map', 'video_map',
                       'main_map', 'audio_ap_sp', 'video_ap_sp', 'main_ap_sp', 'audio_ap_ns', 'video_ap_ns',
                       'main_ap_ns', 'audio_auc', 'video_auc', 'main_auc', 'orig_main_map', 'orig_video_map', 'orig_audio_map']
            keys_remove = ['y_true', 'y_audio_class_ids', 'y_video_class_ids', 'y_main_class_ids']

            results = []
            for weights_file in weights_files:
                # Set weights
                model.load_weights(weights_file)

                # Get results and append
                result = evaluate(model, weights_file, test_gen, test_ann_dir)

                # Remove unnecessary pairs
                for k in keys_remove:
                    del result[k]

                # Add weights name to results and move to the start of the OrderedDict
                result['weights'] = weights_file
                result.move_to_end('weights', last=False)

                results.append(result)

            file_name = 'evaluation-results.csv'
            save_csv(results, columns, file_name)
            print(f"Saved evaluation results to: {file_name}")

    # Print duration
    end = timer()
    duration = end - start
    print(f"Duration: {datetime.timedelta(seconds=duration)}")


if __name__ == "__main__":
    main()
