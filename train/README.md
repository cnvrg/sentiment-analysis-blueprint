#  cnvrg sentiment_analysis_train

Notes for this Component

## How It Works

The library trains a sentiment analysis model on given labeled data and produces a prediction model and a tokenizer.
By default the library needs the receive a single path (--filename) for a local file.
The library splits the content of the file to train and test set 80% train and 20% test.


## How To Run

python3 sentiment_analysis.py --input_filename [YOUR_LABLED_DATA_FILE]

run python3 train.py -f  for info about more optional parameters
                                     
## Parameters

`--input_filename` - (String) (Required param) Path to a local labeled data file which contains the data that is used for training and validation.

`--output_token_file` - (String) (Default: 'tokenizer.pickle') Filename for saving the tokenizer.

`--output_model_file` - (String) (Default: 'model.h5') Filename for saving the model.

`--text_column` - (String) (Default: 'text') Name of text column in dataframe.

`--label_column` - (String) (Default: 'sentiment') Name of label column in dataframe.

`--epochs_val` - (int) (Default: 5) The number of epochs the algorithm performs in the training phaze.

`--batch_size_val` - (int) (Default: 256) The size of each batch to train on.

`--w2v_epochs_val` - (int) (Default: 32) The number of training epochs to run for word to vector model.

`--train_size` - (float) (Default: 0.8) The fraction of dataset to be assigned as training data.

`--w2v_size` - (int) (Default: 300) The vector size parameter in word to vector model.

`--w2v_window` - (int) (Default: 7) The window size parameter in word to vector model.

`--w2v_min_count` - (int) (Default: 10) The min count parameter in word to vector model.

`--seq_len` - (int) (Default: 300) The seq len for NLP model.

`--local_dir` - (string) (Default: cnvrg_workdir) The file_dir to store model file to. 
