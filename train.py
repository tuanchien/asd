# Author: Tuan Chien, James Diprose

import datetime
import os
import pathlib
import secrets
from timeit import default_timer as timer

import click
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from ava_asd.config import get_optimiser, get_model, get_loss_weights, read_config
from ava_asd.generator import AvGenerator, DatasetSubset
from ava_asd.telegrambot import UpdateBot
from ava_asd.utils import set_gpu_memory_growth


def get_callbacks(data_path, sess_id, config, bot_config_file):
    """
    Get a list of callbacks to use for training.
    """

    # Get config values
    mode = config['mode']
    tb_logdir = config['tb_logdir']
    save_best_only = config['save_best_only']
    use_earlystopping = config['use_earlystopping']
    es_patience = config['es_patience']

    callbacks = []

    # Model checkpoint
    model_file_pattern = sess_id + '-' + mode + '-weights-{epoch:02d}-{val_main_out_accuracy:.4f}.hdf5'
    experiment_path = os.path.join(data_path, 'experiments', sess_id)
    pathlib.Path(experiment_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(experiment_path, model_file_pattern)
    callbacks.append(ModelCheckpoint(model_path, monitor='val_main_out_accuracy', verbose=1,
                                     save_best_only=save_best_only, mode='max'))

    # Tensorboard
    tb_session_dir = os.path.join(tb_logdir, sess_id)  # Puts the results in a unique TensorBoard session
    pathlib.Path(tb_session_dir).mkdir(parents=True, exist_ok=True)
    callbacks.append(TensorBoard(log_dir=tb_session_dir, update_freq='batch'))

    # Early stopping
    if use_earlystopping:
        es_patience = es_patience
        callbacks.append(EarlyStopping(monitor='val_main_out_loss', patience=es_patience))

    # Telegram reporting bot
    if bot_config_file is not None:
        bot_config = read_config(bot_config_file.name)
        callbacks.append(UpdateBot.from_dict(bot_config, sess_id=sess_id))

    return callbacks


@click.command()
@click.argument('config-file', type=click.File('r'))
@click.argument('data-path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--bot-config', type=click.File(), default=None)
def main(config_file, data_path, bot_config):
    """ Train the audio visual model.

    CONFIG_FILE: the config file with settings for the experiment.
    DATA_PATH: the path to the folder with the data files.
    """

    # Start time for measuring experiment
    start = timer()

    # Enable memory growth on GPU
    set_gpu_memory_growth(True)

    # Read configs
    config = read_config(config_file.name)

    # Load model
    model, loss = get_model(config)

    # Load data generators
    train_gen = AvGenerator.from_dict(data_path, DatasetSubset.train, config)
    test_gen = AvGenerator.from_dict(data_path, DatasetSubset.valid, config)

    print(train_gen)
    print(test_gen)

    # Create list of callbacks to use for training
    sess_id = secrets.token_urlsafe(5)  # Create session id
    callbacks = get_callbacks(data_path, sess_id, config, bot_config)
    callbacks.append(train_gen)
    callbacks.append(test_gen)

    # Make optimiser and get loss weights
    optimiser = get_optimiser(config)
    loss_weights = get_loss_weights(config)

    # Compile model
    model.compile(loss=loss, optimizer=optimiser, metrics=['accuracy'], loss_weights=loss_weights)

    # Dump a summary
    model.summary()

    # Run training
    epochs = config['epochs']
    model.fit(train_gen.dataset, epochs=epochs, validation_data=test_gen.dataset, callbacks=callbacks)

    # Print duration
    end = timer()
    duration = end - start
    print(f"Duration: {datetime.timedelta(seconds=duration)}")


if __name__ == "__main__":
    main()
