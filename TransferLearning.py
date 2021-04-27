import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
import torch.nn as nn
from collections import defaultdict
from matplotlib import pyplot as plt
torch.cuda.empty_cache()

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# class FinNewsDataset(Dataset):
#     def __init__(self, corpora, sentiment, tokenizer):
#         self.corpora = corpora
#         self.sentiment = sentiment
#         self.tokenizer = tokenizer
#         self.max_len = MAX_LEN
#
#     def __len__(self):
#         return len(self.corpora)
#
#     def __getitem__(self, item):
#         article = str(self.corpora[item])
#         target = self.sentiment[item]
#         encoding = self.tokenizer.encode_plus(article, add_special_tokens=True, max_length=self.max_len,
#                                               return_token_type_ids=False, pad_to_max_length=True,
#                                               return_attention_mask=True, return_tensors='pt')
#
#         return {'finance_text': article, 'input_ids': encoding['input_ids'].flatten(),
#                 'attention_mask': encoding['attention_mask'].flatten(),
#                 'targets': torch.tensor(target, dtype=torch.long)}
#
# def generate_dataloader(df, tokenizer):
#     ds = FinNewsDataset(corpora=df[SEQUENCE_COL].to_numpy(), sentiment=df[SENTIMENT].to_numpy(), tokenizer=tokenizer)
#     return DataLoader(ds, batch_size=BATCH_SIZE,  num_workers=0)

class FinNewsDataset(Dataset):
    """
    This function will create an iterable instance of
    data that will be used in out custom DataLoader
    """
    def __init__(self, corpora, sentiment, tokenizer):
        self.corpora = corpora
        self.sentiment = sentiment
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.corpora)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(self.corpora[item], add_special_tokens=True, max_length=MAX_LEN,
                                              return_token_type_ids=False, pad_to_max_length=True,
                                              return_attention_mask=True, return_tensors='pt')

        data_loader_item = {'finance_text': self.corpora[item], 'input_ids': encoding['input_ids'].flatten(),
                           'attention_mask': encoding['attention_mask'].flatten(),
                           'targets': torch.tensor(self.sentiment[item], dtype=torch.long)}
        return data_loader_item


def generate_dataloader(df, tokenizer):
    ds = FinNewsDataset(corpora=df[SEQUENCE_COL].to_numpy(), sentiment=df[SENTIMENT].to_numpy(), tokenizer=tokenizer)
    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL, num_labels=n_classes)
        self.drop = nn.Dropout(p=DROPOUT)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(return_dict=False, input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def format_split_data(data):
    df_train, df_test = train_test_split(data, test_size=0.10, random_state=RANDOM_SEED)
    df_train, df_val = train_test_split(df_train, test_size=0.35, random_state=RANDOM_SEED)
    train_data_loader = generate_dataloader(df_train, tokenizer)
    val_data_loader = generate_dataloader(df_val, tokenizer)
    test_data_loader = generate_dataloader(df_test, tokenizer)
    return df_train, df_train, df_val, train_data_loader, val_data_loader, test_data_loader


if __name__ == '__main__':
    BATCH_SIZE = 16
    RANDOM_SEED = 402
    LR = 2e-5
    DROPOUT = 0.3
    PRETRAINED_MODEL = 'bert-base-cased' #'distilbert-base-uncased' #'bert-base-cased' #'bert-base-uncased'
    MAX_LEN = 150
    EPOCHS = 10
    SEQUENCE_COL = 'Headline'
    SENTIMENT = 'Sentiment'
    df = pd.read_csv('headlines_labeled_augmented.csv')  #'Data/headlines_labeled 2.csv').tail(1000)  #'Data/hybrid_deep_curated_data.csv')  # #'Data/hybrid_cureted_data.csv') #
    df = df[[SEQUENCE_COL, SENTIMENT]]
    class_names = df[SENTIMENT].unique().tolist()
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL, model_max_length=MAX_LEN)

# 888-437-7611

    df_train, df_test, df_val, train_data_loader, val_data_loader, test_data_loader = format_split_data(df)
    device = get_device()
    model = SentimentClassifier(len(class_names))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print('=' * 15)
        print('Epoch {0}/{1}'.format(epoch + 1, EPOCHS))
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        print('Train Loss {0} Accuracy {1}'.format(train_loss, train_acc))
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
        print('Val loss {val_loss} accuracy {val_acc}')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # axes[0].plot(history['train_acc'], history['val_acc'])
    # axes[1].plot( history['val_acc'], history['val_loss'])
    # fig.tight_layout()
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()
    test_acc, _ = eval_model(model, test_data_loader, loss_fn, device,len(df_test))
    print(test_acc)