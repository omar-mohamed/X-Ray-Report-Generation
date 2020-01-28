import tensorflow as tf
from .bahdanau_attention import BahdanauAttention
from utility import get_layers

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size,output_layers, pretrained_embeddings=None):
        super(RNN_Decoder, self).__init__()
        self.units = units
        if pretrained_embeddings is not None:

            self.embedding = tf.keras.layers.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1],
                                                       weights=[pretrained_embeddings]
                                                       ,trainable=True)

        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size+1, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       stateful=True,
                                       recurrent_initializer='glorot_uniform')

        # self.gru2 = tf.keras.layers.GRU(self.units,
        #                                 return_sequences=True,
        #                                 return_state=True,
        #                                 recurrent_initializer='glorot_uniform')

        self.fc1 = tf.keras.layers.Dense(self.units,activation='relu')
        self.output_layers=get_layers(output_layers, 'relu')
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output, state = self.gru2(output)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        for layer in self.output_layers:
            x = layer(x)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def get_zero_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

    def get_hidden_state(self):
        return self.gru.states

    def set_hidden_state(self,states):
        self.gru.reset_states(states=states)

    def reset_hidden_state(self,batch_size):
        self.gru.reset_states(states=[self.get_zero_state(batch_size)])


