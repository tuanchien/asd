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

# Telegram bot for reporting results after every epoch.
# Author: Tuan Chien, James Diprose

from socket import gethostname

import requests
from tensorflow.keras.callbacks import Callback


class UpdateBot(Callback):
    def __init__(self, api_key: str, chat_id: str, sess_id: str = ''):
        super(Callback, self).__init__()

        api_url = 'https://api.telegram.org/bot'
        self.url_prefix = api_url + api_key + '/'
        self.chat_id = chat_id
        self.machine = gethostname()
        self.sess_id = sess_id

    def on_epoch_end(self, epoch, logs={}):
        loss = logs['loss']
        a_out_accuracy = logs['a_out_accuracy']
        v_out_accuracy = logs['v_out_accuracy']
        main_out_accuracy = logs['main_out_accuracy']
        val_a_out_accuracy = logs['val_a_out_accuracy']
        val_v_out_accuracy = logs['val_v_out_accuracy']
        val_main_out_accuracy = logs['val_main_out_accuracy']

        msg = f'[{self.machine}] sess_id: {self.sess_id}\n'
        msg += f'Epoch {epoch + 1}\n\n'
        msg += 'Training:\n'
        msg += f'loss: {loss:.4f}\n'
        msg += f'a_acc: {a_out_accuracy:.4f} v_acc: {v_out_accuracy:.4f}\n'
        msg += f'av_acc: {main_out_accuracy:.4f}\n\n'
        msg += 'Validation:\n'
        msg += f'a_acc: {val_a_out_accuracy:.4f} v_acc: {val_v_out_accuracy:.4f}\n'
        msg += f'av_acc: {val_main_out_accuracy:.4f}'
        self.send_msg(msg)

    def send_msg(self, msg):
        try:
            data = {
                'chat_id': self.chat_id,
                'text': msg
            }
            url = self.url_prefix + 'sendMessage'
            requests.post(url, data)
        except Exception as e:
            # Not sure if this is the best thing to do, but I don't want training to crash if my internet cuts out!
            print(f"UpdateBot: exception sending message with Telegram bot: {e}")

    def report_evaluation(self, weights, loss, audio_acc, video_acc, av_acc, auroc, aupr):
        msg = f'[{self.machine}] Model evaluation\n'
        msg += f'Weights: {weights}\n'
        msg += f'Loss: {loss:.4f}\n'
        msg += f'Audio accuracy: {audio_acc:.4f}\n'
        msg += f'Video accuracy: {video_acc:.4f}\n'
        msg += f'AV accuracy: {av_acc:.4f}\n'
        msg += f'Area under ROC (auROC): {auroc:.4f}\n'
        msg += f'Area under Precision-Recall: {aupr:.4f}\n'
        self.send_msg(msg)

    @staticmethod
    def from_dict(config: dict, **kwargs):
        config_ = dict(config)

        # Kwargs is used to override values in the config
        for key, value in kwargs.items():
            config_[key] = value

        assert 'api_key' in config_, "UpdateBot.from_dict: 'api_key' not in config or **kwargs"
        assert 'chat_id' in config_, "UpdateBot.from_dict: 'chat_id' not in config or **kwargs"

        api_key = config_['api_key']
        chat_id = config_['chat_id']
        sess_id = config_.get('sess_id', '')  # Default value

        return UpdateBot(api_key, chat_id, sess_id)
