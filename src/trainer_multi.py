import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

from bert.bert_model import BERT_model
import logging
from util import logger

def get_data(size=None):
    #df = pd.read_csv('data/train.tsv', delimiter='\t', header=None)
    df = pd.read_csv('data/kn/kannada-news-dataset/train.csv', header=None)
    
    if size:
        df = df[:size]
    print('Trainig data:\n',df.shape)
    print('Class distribution:\n', df[1].value_counts())
    return df

def get_max_sent_length(tokenized_df):
    max_len = 0
    for i in tokenized_df.values:
        if len(i) > max_len:
            max_len = len(i)
    return max_len

def get_padded_and_attention_mask_ndarray(tokenized_df, max_len):
    padded_ndarray = np.array([i + [0]*(max_len-len(i)) for i in tokenized_df.values])
    print('Padded array:\n', padded_ndarray.shape)
    
    attention_mask_ndarray = np.where(padded_ndarray != 0, 1, 0)
    print('Attention mask:\n', attention_mask_ndarray.shape)
    
    return (padded_ndarray, attention_mask_ndarray)
    
def print_test_metrics(act, pred):
    print('Confusion Matrix:')
    print(confusion_matrix(act, pred))
    print(classification_report(act, pred))

def sentence_classifier(bert_feature_array, target_labels, TEST_RATIO=0.1):
    # Split train and test
    train_features, test_features, train_labels, test_labels = train_test_split(bert_feature_array, target_labels,
    test_size=TEST_RATIO)

    # Train the model
    classifier = LogisticRegression()
    classifier.fit(train_features, train_labels)
    print('Training complete')

    # Get the test results
    pred_labels = classifier.predict(test_features) 
    #print('Pred:', pred_labels)
    # Print the evaluation results
    print_test_metrics(test_labels, pred_labels)

def trainer():
    # Load the TSV training data
    df = get_data(size=50)

    # Load the tokenizer and BERT models
    bert = BERT_model()
    bert.load_BERT(small=True)

    # Tokenize the sentences in the training data
    tokenized_df = df[0].apply(lambda sent: bert.tokenize_sentence(sent))
    MAX_LEN = get_max_sent_length(tokenized_df)
    print('Maximum sentence length:', MAX_LEN)

    # Pad and get the attention mask arrays
    padded_ndarray, attention_mask_ndarray = get_padded_and_attention_mask_ndarray(tokenized_df, MAX_LEN)

    # Feed the padded array and attention mask array to the bert model and get the hidden states from output layer of BERT
    bert_hidden_states = bert.get_bert_embeddings(padded_ndarray, attention_mask_ndarray, batchsize=500)
    # TODO Save the features and target lables

    # Slice the hidden states of shape (number of training examples, max number of tokens=MAX_LEN, number of hidden units in BERT=768)
    # And take only the CLS output of the BERT
    bert_feature_array = bert_hidden_states[:,0,:].numpy()
    
    print('Bert features shape:\n', bert_feature_array.shape)
    # The target labels
    target_labels = df[1]

    # Train a classifier
    sentence_classifier(bert_feature_array, target_labels)

if __name__ == "__main__":
    trainer()
    
