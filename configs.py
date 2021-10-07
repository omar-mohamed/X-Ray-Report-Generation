from tags import tags


class argHandler(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {'help, --h, -h': 'show this super helpful message and exit'}
    def setDefaults(self):
        self.define('train_csv', './IU-XRay/training_set.csv',
                    'path to training csv containing the images names and the labels')
        self.define('test_csv', './IU-XRay/testing_set.csv',
                    'path to testing csv containing the images names and the labels')
        self.define('all_data_csv', './IU-XRay/all_data.csv',
                    'path to all data csv containing the images names and the labels')
        self.define('image_directory', './IU-XRay/images',
                    'path to folder containing the patient folders which containing the images')
        self.define('output_images_folder', './output_images',
                    'path to folder containing output images')
        self.define('data_dir', './IU-XRay',
                    'path to folder containing the patient folders which containing the images')
        self.define('visual_model_name', 'fine_tuned_chexnet',
                    'path to folder containing the patient folders which containing the images')
        self.define('visual_model_pop_layers', 1,
                    'number of conv layers to pop to get visual features')
        self.define('embedding_dim', 400,
                    'size of embedding vector')
        self.define('csv_label_columns', ['Caption'], 'the name of the label columns in the csv')
        self.define('image_target_size', (224, 224, 3), 'the target size to resize the image')

        self.define('max_sequence_length', 170,
                    'Maximum number of words in a sentence')
        self.define('num_epochs', 100, 'maximum number of epochs')
        self.define('units', 1024, 'number of units in the decoder')
        self.define('tags_reducer_units', 1024, 'number of units to reduce the embeddings. choose 0 if you want to keep it as is')
        self.define('encoder_layers', [0.4], 'a list describing the hidden layers of the encoder. Example [10,0.4,5] will create a hidden layer with size 10 then dropout wth drop prob 0.4, then hidden layer with size 5. If empty it will connect to output nodes directly.')
        self.define('classifier_layers', [0.4], 'a list describing the hidden layers of the encoder. Example [10,0.4,5] will create a hidden layer with size 10 then dropout wth drop prob 0.4, then hidden layer with size 5. If empty it will connect to output nodes directly.')
        self.define('tags_attention', True, 'if use attention on tags embeddings with visual features')

        self.define('tokenizer_vocab_size', 1501,
                    'The number of words to tokinze, the rest will be set as <unk>')
        self.define('batch_size', 15, 'batch size for training and testing')
        self.define('generator_workers', 4, 'The number of cpu workers generating batches.')
        self.define('generator_queue_length', 16, 'The maximum number of batches in the queue to be trained on.')
        self.define('tags_threshold', 0.1,
                    'The threshold from which to detect a tag.')
        self.define('ckpt_path', './checkpoints/',
                    'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
        self.define('continue_from_last_ckpt', True,
                    'continue training from last ckpt or not')
        self.define('epochs_to_evaluate', 3, 'The number of epochs to train before evaluating on the test set.')
        self.define('beam_width', 7, 'The beam search width during evaluation')
        self.define('learning_rate', 1e-3, 'The optimizer learning rate')
        self.define('optimizer_type', 'Adam', 'Choose from (Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam)')
        self.define('tags', tags,
                    'the names of the tags')

    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    def help(self):
        print('Arguments:')
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print('  --' + item + (' ' * currentSpacing) + self._descriptions[item])
        exit()
