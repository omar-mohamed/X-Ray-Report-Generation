import tensorflow as tf
import numpy as np


class BeamPaths():
    def __init__(self, k):
        self.paths = []
        self.ended_paths = []
        self.k = k

    def add_path(self, path):
        if path.ended():
            self.ended_paths.append(path)
        else:
            self.paths.append(path)

    def get_best_path(self):
        return self.paths[0]

    def get_ended_paths(self):
        self.ended_paths.sort(key=lambda x: x.get_total_probability())
        return self.ended_paths

    def get_ended_paths_count(self):
        return len(self.ended_paths)

    def get_running_paths_count(self):
        return len(self.paths)

    def sort(self, remove_extra=False):
        self.paths.sort(key=lambda x: x.get_total_probability())
        if remove_extra:
            self.paths = self.paths[0:self.k]

    def add_top_k_paths(self, paths):
        paths.sort(key=lambda x: x.get_total_probability())
        for i in range(self.k):
            self.add_path(paths[i])

    def get_best_k(self):
        return self.paths[0:self.k]

    def clear(self):
        self.paths = []

    def pop_best_k(self):
        self.paths = self.paths[self.k:]

    def should_stop(self):
        if len(self.paths) == 0:
            return True
        if len(self.ended_paths) == 0:
            return False
        self.sort()
        return self.get_ended_paths()[0].get_total_probability() <= self.paths[0].get_total_probability()

    def get_best_paths_input(self):
        tokens = []
        size = min(len(self.paths), self.k)

        for i in range(size):
            tokens.append(self.paths[i].get_last_token())
        return tf.expand_dims(tokens, 1)

    def get_best_paths_hidden(self):
        hidden_size = self.paths[0].get_hidden_size()
        size = min(len(self.paths), self.k)
        hidden = np.zeros(shape=(size, hidden_size[1]), dtype=np.float32)
        for i in range(size):
            hidden[i] = self.paths[i].get_hidden_layer()[0]

        return tf.convert_to_tensor(hidden)
