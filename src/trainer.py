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
    logging.info('Trainig data:{}'.format(df.shape))
    logging.info('Class distribution:\n{}'.format(df[1].value_counts()))
    return df

def get_max_sent_length(tokenized_df):
    max_len = 0
    for i in tokenized_df.values:
        if len(i) > max_len:
            max_len = len(i)
    return max_len
    
def print_test_metrics(act, pred):
    logging.info('Confusion Matrix:\n' + str(confusion_matrix(act, pred)))
    logging.info('Report:\n' + str(classification_report(act, pred)))

def sentence_classifier(bert_feature_array, target_labels, TEST_RATIO=0.1):
    # Split train and test
    train_features, test_features, train_labels, test_labels = train_test_split(bert_feature_array, target_labels,
    test_size=TEST_RATIO)

    # Train the model
    classifier = LogisticRegression()
    classifier.fit(train_features, train_labels)
    logging.info('Training complete')

    # Get the test results
    pred_labels = classifier.predict(test_features) 
    # logging.info('Pred:{}'.format(pred_labels))
    # Print the evaluation results
    print_test_metrics(test_labels, pred_labels)

def trainer():
    # Load the TSV training data
    df = get_data()

    # Load the tokenizer and BERT models
    bert = BERT_model()
    bert.load_BERT(small=True)

    # Tokenize the sentences in the training data
    tokenized_df = df[0].apply(lambda sent: bert.tokenize_sentence(sent))
    MAX_LEN = get_max_sent_length(tokenized_df)
    logging.info('Maximum sentence length:{}'.format(MAX_LEN))

    # Provide the tokenized sentences and get the BERT embeddings back
    bert_hidden_states = bert.convert_tokenized_sent_to_bert_emb(tokenized_df, MAX_LEN, batch_size=500)
    # TODO Save the features and target lables

    # Slice the hidden states of shape (number of training examples, max number of tokens=MAX_LEN, number of hidden units in BERT=768)
    # And take only the CLS output of the BERT
    bert_feature_array = bert_hidden_states[:,0,:].numpy()
    
    logging.info('Bert features shape:{}'.format(bert_feature_array.shape))
    # The target labels
    target_labels = df[1]

    # Train a classifier
    sentence_classifier(bert_feature_array, target_labels)

if __name__ == "__main__":
    trainer()
    
