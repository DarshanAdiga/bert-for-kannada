# ಕನ್ನಡದಲ್ಲಿ ಬರ್ಟ್ ನ ಉಪಯೋಗ (BERT for KANNADA)
Easy-to-start code base to make use of BERT for kannada language.

## ಕನ್ನಡ ವಾಕ್ಯ ವರ್ಗೀಕರಿಸುವ ತಂತ್ರಾಂಶ (Kannada Sentence Classifier)
Used the **pretrained multilingual BERT** to generate sentence embeddings and built a sentence classifier.

### ನಿಖರತೆಯ ಮಾಪನಗಳು (Accuracy Metrics)

|               |precision    |recall  |f1-score   |support|
|---------------|-------------|--------|-----------|-------|
|entertainment  |  0.85       |  0.93  |  0.89     |  282  |
|       sports  |  0.85       |  0.79  |  0.82     |  177  |
|         tech  |  0.82       |  0.64  |  0.72     |  58   |

## TODO
* Use of better classifier
* Fine-tune the BERT model on larger corpus
* Use BERT for other NLP tasks in Kannada

## References
https://github.com/huggingface/transformers
https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb
