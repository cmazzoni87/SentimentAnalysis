import numpy as np
import pandas as pd
import torch

from datetime import datetime
from config import ModelParamsConfig as vals
from config import get_device
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader,TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from typing import Tuple
from tqdm import tqdm


MODEL_SAVE_PATH = vals.MODEL_PATH
MODEL_NAME = vals.MODEL_NAME
BATCH_SIZE = vals.BATCH_SIZE
LR = vals.LR
MAX_LEN = vals.MAX_LEN
EPOCHS = vals.EPOCHS
LABEL_NUM = vals.LABEL_NUM
TRAIN_TEST_SPLIT = vals.TRAIN_TEST_SPLIT
device = get_device()


def evaluate(dataloader_val, model) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in dataloader_val:
        batch = tuple(b.long() for b in batch)
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def f1_score_func(preds, labels) -> f1_score:
    """
    :param preds: models prediction
    :param labels: actual values
    :return: f1_score int
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def guesser_plus(article_text, tokenizer, classes) -> str:
    # prepare our text into tokenized sequence
    inputs = tokenizer(article_text, padding=True, truncation=True, max_length=50, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return classes[probs.argmax()]


def generate_dataloader(input_x, input_y, tokenizer, shuffle) -> DataLoader:
    """
    :param input_x: Data Array
    :param input_y: Labels Array
    :param tokenizer: Respective tokenizer from model used
    :param shuffle: Bool, shuffle whether or not to change the ordering of the data
    :return: DataLoader used to train model
    """
    encoded_data = tokenizer.batch_encode_plus(input_x,
                                               add_special_tokens=True,
                                               return_attention_mask=True,
                                               padding=True,
                                               max_length=MAX_LEN,
                                               return_tensors='pt')
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(input_y.values)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=BATCH_SIZE)
    return dataloader


def train_model(file_path) -> None:
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME,
                                                          num_labels=LABEL_NUM,
                                                          output_attentions=False,
                                                          output_hidden_states=False,
                                                          attention_probs_dropout_prob=0.2,
                                                          hidden_dropout_prob=0.2)
    sentiment_label = 'sentiment'
    text_label = 'generated text'

    data = pd.read_csv(file_path)
    data[text_label] = data[text_label].str.lower()
    test_df = data.sample(frac=0.05, random_state=1)

    input_data = data.drop(test_df.index)
    encoder = LabelEncoder()
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    input_data['encoded_sentiment'] = encoder.fit_transform(input_data[sentiment_label])
    x_train, x_val, y_train, y_val = train_test_split(input_data[text_label],
                                                      input_data['encoded_sentiment'],
                                                      test_size=TRAIN_TEST_SPLIT)

    dataloader_train = generate_dataloader(x_train, y_train, tokenizer, True)
    dataloader_validation = generate_dataloader(x_val, y_val, tokenizer, False)

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-4)
    steps = len(dataloader_train) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    model.to(device)

    for epoch in range(EPOCHS):
        print('=' * 15)
        print('Epoch {0}/{1}'.format(epoch + 1, EPOCHS))
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train,
                            desc='Epoch {0}/{1}'.format(epoch + 1, EPOCHS),
                            leave=False,
                            disable=False)
        for batch in  progress_bar: #  dataloader_train:
            model.zero_grad()
            batch = tuple(b.long() for b in batch)

            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[2].to(device)}
            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        loss_train_avg = loss_train_total / len(dataloader_train)
        val_loss, _, _ = evaluate(dataloader_validation, model)

        print('Train Loss {0}'.format(loss_train_avg))
        print('Validation Loss {0}'.format(val_loss))

    val_loss, predictions, true_vals = evaluate(dataloader_validation, model)
    val_f1 = f1_score_func(predictions, true_vals)
    if val_f1 > 0.95:
        print('Model has a high F1 score, it will be saved.')
        torch.save(model.state_dict(), 'sentiment_model_from_augmented_{}.bin'.format(d))

    print('Val Loss = ', val_loss)
    print('Val F1 = ', val_f1)

    encoded_classes = encoder.classes_
    predicted_category = [encoded_classes[np.argmax(x)] for x in predictions]
    true_category = [encoded_classes[x] for x in true_vals]
    x = 0
    for i in range(len(true_category)):
        if true_category[i] == predicted_category[i]:
            x += 1

    print('Accuracy Score = ', x / len(true_category))

    test_df['Prediction'] = test_df['text'].apply(guesser_plus, tokenizer=tokenizer, classes=['Negative', 'Positive'])
    test_df['Correct Prediction'] = test_df['Prediction'] == test_df['sentiment']
    prediction_result = test_df.groupby('Correct Prediction')['text'].count()
    print('Prediction Results')
    print(prediction_result)