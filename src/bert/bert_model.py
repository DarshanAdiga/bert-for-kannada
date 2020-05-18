# import numpy as np
import torch
import transformers as ppb

import warnings
import time
warnings.filterwarnings('ignore')
import logging
from util import logger

class BERT_model():
    def __init__(self):
        self.is_loaded = False

    def load_BERT(self, small=True):
        """Loads the pretrained BERT models and the corresponding Tokenizers

        Keyword Arguments:
            small {bool} -- Whether to load the smaller model or the bigger one (default: {True})

        Returns:
            [tuple] -- Pretrained DistilBERT or BERT and the corresponding Tokenizer
        """

        if self.is_loaded:
            logging.warning('BERT is already loaded!')

        if small:
            # For DistilBERT:
            model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-multilingual-cased')
        else:
            # For BERT
            model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-multilingual-cased')

        # Load pretrained model/tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        self.is_loaded = True
        logging.info('BERT has been loaded successfully')
        
    def tokenize_sentence(self, sent):
        """Tokenizes the given sentence sent

        Arguments:
            sent {str} -- Sentence to be tokenized

        Returns:
            list -- List of tokens
        """
        # TODO take batch of sentences
        return self.tokenizer.encode(sent, add_special_tokens=True)

    def get_bert_embeddings(self, padded_feature_array, attention_mask_array, batchsize=-1):
        """Runs the BERT model on the given padded_feature_array and attention_mask_array to get the embeddings.

        Arguments:
            padded_feature_array {numpy.ndarray} -- Tokenized and padded sentence features  
            attention_mask_array {numpy.ndarray} -- The mask indicating the position of sentence features

        Keyword Arguments:
            batchsize {int} -- To avoid memory errors, process the sentences in batches (default: {-1})

        Returns:
            torch.tensor -- Returns torch.tensor containing BERT output weights as embeddings
        """
        logging.info('Going to get BERT embeddings for {} records'.format(padded_feature_array.shape[0]))
        if batchsize > 0:
            # Generate embeddings batch-by-batch
            logging.info('Running batch-wise. Original shape:{}'.format(padded_feature_array.shape))
            # Limit the max batch size
            max_size = padded_feature_array.shape[0]
            batchsize = max_size if batchsize > padded_feature_array.shape[0] else batchsize
            final_embeddings = None
            logging.debug('-' * 30)
            for b_start in range(0, max_size, batchsize):
                b_end = b_start + batchsize
                b_num = int(b_start/batchsize)
                logging.debug('Batch {} start:{}, end:{}'.format(b_num, b_start, b_end))
                batch_sents = padded_feature_array[b_start:b_end]
                batch_attentions = attention_mask_array[b_start:(b_start + batchsize)]
                # Get the embeddings for this batch
                batch_embeddings_tensor = self.__get_embeddings_per_batch(batch_sents, batch_attentions)
                logging.debug('Batch emb size:{}'.format(batch_embeddings_tensor.size()))
                # Accumulate the embeddings generated so far
                if final_embeddings == None:
                    final_embeddings = batch_embeddings_tensor
                else:
                    final_embeddings = torch.cat([final_embeddings, batch_embeddings_tensor], dim=0)
                    logging.info('Accumulated emb size:{}'.format(final_embeddings.size()))
                # End of for
                logging.debug('-' * 30)
            # Return the final embeddings
            return final_embeddings
        else:
            return self.__get_embeddings_per_batch(padded_feature_array, attention_mask_array)

    def __get_embeddings_per_batch(self, padded_feature_array, attention_mask_array):
        input_token_id_matrix = torch.tensor(padded_feature_array)
        attention_mask = torch.tensor(attention_mask_array)
        # No training of BERT now!
        with torch.no_grad():
            t0 = time.time()
            output_hidden_states = self.model(input_token_id_matrix, attention_mask=attention_mask)
            output_hidden_states = output_hidden_states[0]
            t1 = time.time()
            logging.info('Time taken:{} seconds'.format(int(t1-t0)))
            return output_hidden_states
