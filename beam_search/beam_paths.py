import tensorflow as tf
import numpy as np
class BeamPaths():
    def __init__(self, k):
        self.paths=[]
        self.ended_paths=[]
        self.k = k


    def add_path(self,path):
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

    def sort(self, remove_extra=True):
        self.paths.sort(key=lambda x: x.get_total_probability())
        if remove_extra:
            self.paths=self.paths[0:self.k]

    def get_best_k(self):
        return self.paths[0:self.k]

    def clear(self):
        self.paths=[]

    def get_best_paths_input(self):
        tokens=[]
        for i in range(len(self.paths)):
            tokens.append(self.paths[i].get_last_token())
        return tf.expand_dims(tokens, 1)

    def get_best_paths_hidden(self):
        hidden_size=self.paths[0].get_hidden_size()
        hidden = np.zeros(shape=(len(self.paths), hidden_size[1]))
        for i in range(len(self.paths)):
            hidden[i]=self.paths[i].get_hidden_layer()[0]

        return hidden