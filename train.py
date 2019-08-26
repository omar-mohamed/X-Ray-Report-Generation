import tensorflow as tf
from models.CNN_encoder import CNN_Encoder
from models.RNN_decoder import RNN_Decoder
from chexnet_wrapper import ChexnetWrapper
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from utility import get_sample_counts
import time
from tensorflow.python.eager.context import eager_mode, graph_mode
from augmenter import augmenter
from medical_w2v_wrapper import Medical_W2V_Wrapper
from tokenizer_wrapper import TokenizerWrapper
import matplotlib.pyplot as plt

config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
class_names = cp["DEFAULT"].get("class_names").split(",")
image_source_dir = cp["DEFAULT"].get("image_source_dir")
data_dir = cp["DEFAULT"].get("data_dir")

# train config
image_dimension = cp["TRAIN"].getint("image_dimension")

# test config
BATCH_SIZE = cp["TEST"].getint("batch_size")
steps = cp["TEST"].get("test_steps")
training_counts = get_sample_counts(data_dir, "training_set", class_names)
# compute steps
if steps == "auto":
    steps = int(training_counts / BATCH_SIZE)
else:
    try:
        steps = int(steps)
    except ValueError:
        raise ValueError(f"""
            test_steps: {steps} is invalid,
            please use 'auto' or integer.
            """)
print(f"** train_steps: {steps} **")

print("** load test generator **")
max_sequence_length=170
tokenizer_wrapper=TokenizerWrapper(os.path.join(data_dir, "all_data.csv"),class_names[0],max_sequence_length)

data_generator = AugmentedImageSequence(
    dataset_csv_file=os.path.join(data_dir, "training_set.csv"),
    class_names=class_names,
    tokenizer_wrapper=tokenizer_wrapper,
    source_image_dir=image_source_dir,
    batch_size=BATCH_SIZE,
    target_size=(image_dimension, image_dimension),
    augmenter=augmenter,
    steps=steps,
    shuffle_on_epoch_end=True,
)

medical_w2v=Medical_W2V_Wrapper()
embeddings=medical_w2v.get_embeddings_matrix_for_words(tokenizer_wrapper.get_word_tokens_list())
print(embeddings.shape)
del medical_w2v
# image, y = test_sequence.__getitem__(4)
#
# visual_features=get_visual_features(image)
# print(visual_features.shape)


BUFFER_SIZE = 1000
embedding_dim = 400
units = 512
vocab_size = tokenizer_wrapper.get_tokenizer_word_index()
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 1024
attention_features_shape = 64

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size,embeddings)

# w=decoder.get_embedding_weights()
# decoder.set_embedding_weights(embeddings)
# w=decoder.get_embedding_weights()

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


loss_plot = []

@tf.function
def train_step(img_tensor, target):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer_wrapper.get_token_of_word("startseq")] * BATCH_SIZE, 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        # print("encoded")

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
    # print("decoded")
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    # print("will apply gradients")
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    # print("applied gradients")
    return loss, total_loss


with graph_mode():
    chexnet = ChexnetWrapper()


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))

EPOCHS = 100
for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for batch in range(data_generator.steps):
        img, target = data_generator.__getitem__(batch)
        with graph_mode():
            img_tensor = chexnet.get_visual_features(img)

        print("batch: {}".format(batch))
        # img_tensor=np.random.randint(low=-1,high=1,size=(1,1024))
        # img_tensor=np.float32(img_tensor)
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / data_generator.steps)

    if epoch % 1 == 0:
        ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                        total_loss / data_generator.steps))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.savefig("loss.png")

# plt.show()
