import unittest
import pandas
import os, shutil, sys
import yaml
from yaml.loader import SafeLoader
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from train import read_dataset, decode_sentiment, preprocess, split_train_test_data, create_documents, word_2_vector, tokenize_text, padding_sequences, label_encoding, prepare_labels, create_embedding_layer, build_model, compile_model, get_callbacks, fit_model, get_score, get_performace_metrics, save_tokenizer, save_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YAML_ARG_TO_TEST = "test_arguments"

class test_train(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        ''' Set up unittesting parameters '''
        # Parse config file parameters
        cfg_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = cfg_path + "/" + "test_config.yaml"
        self.test_cfg = {}
        with open(cfg_file) as c_info_file:
            self.test_cfg = yaml.load(c_info_file, Loader=SafeLoader)

        self.test_cfg = self.test_cfg[YAML_ARG_TO_TEST]
        self.unittest_dataset_dir = 'unittest_dataset'
        self.unittest_data = 'unittest_data.csv'
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.curr_dir, self.unittest_dataset_dir, self.unittest_data)
        self.df = read_dataset(self.data_path)
        self.train_size_ratio = 0.8
        self.create_docs_data = {
            'text': [self.test_cfg['processed_text']]
        }
        self.create_docs_df = pandas.DataFrame(self.create_docs_data)
        self.w2v_size = 300
        self.w2v_window = 7
        self.w2v_epochs_val = 32
        self.w2v_min_count = 10
        self.seq_len = 300
        self.epochs_val = 64
        self.batch_size_val = 256
        self.output_token_file = "tokenizer.pickle"
        self.output_model_file = "model.h5"

        # Unit-test data directory - temporary 
        self.unittest_dir = "unit_test_data"
        self.local_dir = os.path.dirname(os.path.abspath(__file__))
        self.unittest_data_path = os.path.join(self.local_dir, self.unittest_dir)
        os.mkdir(self.unittest_data_path)

        self.df.target = self.df.target.apply(lambda x: decode_sentiment(x))
        self.stemmer = SnowballStemmer("english")
        self.df.text = self.df.text.apply(lambda x: preprocess(x, self.stemmer))

        self.df_train, self.df_test = split_train_test_data(self.df, self.train_size_ratio)
        self.documents = create_documents(self.df_train)

        # Preparing the model
        self.w2v_model = word_2_vector(self.documents, self.w2v_size, self.w2v_window, self.w2v_epochs_val, self.w2v_min_count)
        self.tokenizer, self.vocab_size = tokenize_text(self.df_train)

        self.x_train, self.x_test = padding_sequences(self.df_train, self.df_test, self.tokenizer, self.seq_len)

        self.encoder, self.labels = label_encoding(self.df_train)
        self.y_train, self.y_test = prepare_labels(self.df_train, self.df_test, self.encoder)

        self.embedding_layer = create_embedding_layer(self.vocab_size, self.tokenizer, self.w2v_model, self.w2v_size, self.seq_len)

        self.model = build_model(self.embedding_layer)
        compile_model(self.model)

        self.callbacks = get_callbacks()

        self.history = fit_model(self.model, self.x_train, self.y_train, self.epochs_val, self.batch_size_val, self.callbacks)
        self.score = get_score(self.model, self.x_test, self.y_test, self.batch_size_val)

        self.acc, self.val_acc, self.loss, self.val_loss = get_performace_metrics(self.history)

        save_tokenizer(self.unittest_data_path , self.output_token_file, self.tokenizer)
        save_model(self.model, self.unittest_data_path, self.output_model_file)

    @classmethod
    def tearDownClass(self):
        ''' Clean up function '''
        shutil.rmtree(self.unittest_data_path)

    def test_df_read(self):
        ''' Checks if pandas dataframe is retuned '''
        self.assertIsInstance(
            self.df, pandas.core.frame.DataFrame
        )

    def test_df_attributes(self):
        ''' Checks the pandas dataframe attributes '''
        self.assertListEqual(
            list(self.df.columns.values), ["target", "timestamp", "datetime", "query", "user", "text"]
        )

    def test_preprocess_text(self):
        ''' Checks pre-processing of texts '''
        self.assertEqual(
            self.test_cfg['processed_text'], preprocess(self.test_cfg['raw_text'], self.stemmer)
        )
      
    def test_sentiment_mapping(self):
        ''' Checks the sentiment mapping function '''
        self.assertEqual(
            "NEGATIVE", decode_sentiment(0)
        )
        self.assertEqual(
            "NEUTRAL", decode_sentiment(2)
        )
        self.assertEqual(
            "POSITIVE", decode_sentiment(4)
        )

    def test_split_train_test_data(self):
        ''' Checks the train and test dataset ratio '''
        self.assertAlmostEqual(
            len(self.df_train) / len(self.df), self.train_size_ratio
        )
        self.assertAlmostEqual(
            len(self.df_test) / len(self.df), 1 - self.train_size_ratio
        )

    def test_docs_creation(self):
        ''' Checks the document creation process '''
        self.assertListEqual(
            create_documents(self.create_docs_df)[0], self.test_cfg['document'] 
        )