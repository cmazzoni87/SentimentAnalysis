# SentimentAnalysis
Sentiment analysis is a form of Natural Language Processing (NLP) that identifies and quantifies text data emotional states and subjective information of the topics, persons and entities within it. Traditionally performed using algorithms such as Naïve Bayes or Support Vector Machines. It's also historically been a supervised task, using labeled datasets to train algorithms, requiring large datasets that are carefully selected, categorized and labeled manually by humans.
Today, the consensus is clear, Deep Learning methods achieve better accuracy on most NLP tasks than other methods. And right now the undisputed Kings of NLP are large pre-trained models called Transformers. Their architecture aim to solve sequence-to-sequence tasks while handling long-range input and output dependencies  with attention and recurrence.

The devil is in the details, we need lots of data and label finance data is hard to come by, thus we are going to build a large dataset based on a small sample of Financial Articles, which we will subsequently be used to build a Sentiment Analysis model that can measure 'sentiment' of financial data.

## Instalation
Download the repo into your machine
Install required packages in requirements.txt
```bash
pip install requirements.txt
```
## main.py
Run main.py, this will run through the entirety of the project, generating data and training the model
```bash
python main.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
