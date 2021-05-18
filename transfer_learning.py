import random
import numpy as np
import pandas as pd
from config import DataProcessorConfig, ModelParamsConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score


def evaluate(dataloader_val):
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



def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
MAX_LEN = 100
EPOCHS = 2
sentiment_label = 'sentiment'
text_lable = 'generated text'

data = pd.read_csv('Data/Full-Economic-News-Processed-Augmented-Syn-Replaced.csv')
test_df = data.sample(frac=0.05, random_state=1)
input_data = data.drop(test_df.index)
encoder = LabelEncoder()
label_number = input_data['sentiment'].unique().shape[0]

input_data['encoded_sentiment'] = encoder.fit_transform(input_data['sentiment'])
x_train, x_val, y_train, y_val = train_test_split(input_data['generated text'],
                                                  input_data['encoded_sentiment'],
                                                  test_size= 0.3)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


encoded_data_train = tokenizer.batch_encode_plus(x_train,
                                                 add_special_tokens=True,
                                                 return_attention_mask=True,
                                                 pad_to_max_length=True,
                                                 max_length=MAX_LEN,
                                                 return_tensors='pt')

encoded_data_val = tokenizer.batch_encode_plus(x_val,
                                               add_special_tokens=True,
                                               return_attention_mask=True,
                                               pad_to_max_length=True,
                                               max_length=MAX_LEN,
                                               return_tensors='pt')

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(y_train.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(y_val.values)


# Pytorch TensorDataset Instance
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=BATCH_SIZE)
dataloader_validation = DataLoader(dataset_val, shuffle=False,  batch_size=BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained(MODEL_NAME,
                                                      num_labels=label_number,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-4)

steps = len(dataloader_train) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
device = torch.device('cuda')

model.to(device)

for epoch in range(EPOCHS):
    print('=' * 15)
    print('Epoch {0}/{1}'.format(epoch + 1, EPOCHS))
    model.train()
    loss_train_total = 0
    for batch in dataloader_train:
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
    print('Train Loss {0}'.format(loss_train_avg))

val_loss, predictions, true_vals = evaluate(dataloader_validation)
val_f1 = f1_score_func(predictions, true_vals)

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