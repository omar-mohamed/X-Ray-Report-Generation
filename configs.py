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
                    'path to folder containing the patient folders which containg the images')
        self.define('output_images_folder', './output_mages',
                    'path to folder containing output images')
        self.define('data_dir', './IU-XRay',
                    'path to folder containing the patient folders which containg the images')
        self.define('visual_model_name', 'fine_tuned_chexnet',
                    'path to folder containing the patient folders which containg the images')
        self.define('embedding_dim', 400,
                    'size of embedding vector')
        self.define('csv_label_columns', ['Findings'], 'the name of the label columns in the csv')
        self.define('image_target_size', (224, 224, 3), 'the target size to resize the image')

        self.define('max_sequence_length', 170,
                    'Maximum number of words in a sentence')
        self.define('num_epochs', 100, 'maximum number of epochs')
        self.define('units', 1024, 'number of units in the decoder')

        self.define('tokenizer_vocab_size', 1001,
                    'The number of words to tokinze, the rest will be set as <unk>')
        self.define('batch_size', 2, 'batch size for training and testing')
        self.define('tags_threshold', 0.3,
                    'The threshold from which to detect a tag.')
        self.define('ckpt_path', './checkpoints/',
                    'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
        self.define('continue_from_last_ckpt', False,
                    'continue training from last ckpt or not')
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
