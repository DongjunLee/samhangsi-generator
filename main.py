#-- coding: utf-8 -*-

import os
from six.moves import cPickle
import tensorflow as tf

from kor_char_rnn.model import Model


flags = tf.app.flags
flags.DEFINE_string('word', '삼행시', 'Input korean word (ex. 삼행시)')
FLAGS = flags.FLAGS



class SamhangSiGenerator:

    def __init__(self):
        self.save_data_path = "kor_char_rnn/save"
        self.session = None

    def load_model(self):
        if self.session is None:
            with open(os.path.join(self.save_data_path, 'config.pkl'), 'rb') as f:
                saved_args = cPickle.load(f)
            with open(os.path.join(self.save_data_path, 'chars_vocab.pkl'), 'rb') as f:
                self.chars, self.vocab = cPickle.load(f)
            self.model = Model(saved_args, training=False)

            self.session = tf.Session()
            init = tf.global_variables_initializer()
            self.session.run(init)

            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_data_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.session, ckpt.model_checkpoint_path)

    def generate_text(self, prime):
        sentence_length = 30
        sample = 1

        result = self.model.sample(self.session, self.chars, self.vocab,
                                   num=sentence_length, prime=prime, sampling_type=sample)
        print(result.split("\n")[0])

def main(args):
    samhangsi_generator = SamhangSiGenerator()
    samhangsi_generator.load_model()
    for char in FLAGS.word:
        samhangsi_generator.generate_text(char)


if __name__ == '__main__':
    tf.app.run()
