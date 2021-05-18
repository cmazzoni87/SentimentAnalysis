from sklearn.metrics import confusion_matrix, classification_report
import torch
from transfer_learning import get_device, SentimentClassifier, eval_model, FinNewsDataset
import pandas as pd
import torch.nn as nn
from seaborn import heatmap
import matplotlib.pyplot as plt


def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["finance_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix):
  hmap = heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')
  plt.show()


def make_predictions():
    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )


if __name__ == '__main__':
    device = get_device()
    SEQUENCE_COL = 'Paraphrased Headlines'
    SENTIMENT = 'Sentiment'
    TEST_FILE_PATH = ''
    PRETRAINED_MODEL = 'bert-base-uncased'
    BATCH_SIZE = 64
    RANDOM_SEED = 402
    LR = 2e-5
    DROPOUT = 0.4
    MAX_LEN = 200
    EPOCHS = 5
    loss_fn = nn.CrossEntropyLoss().to(device)
    df = pd.read_csv('test_data.csv', dtype=str)
    df = df[[SEQUENCE_COL, SENTIMENT]]
    class_names = df[SENTIMENT].unique().tolist()
    model = SentimentClassifier(len(class_names ), PRETRAINED_MODEL, DROPOUT)
    model.load_state_dict(torch.load('best_model_state.bin'))
    model = model.to(device)
    test_data_loader = torch.load('test_dataloader.pth')

    # y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)
    # print(classification_report(y_test, y_pred, target_names=class_names))
    # cm = confusion_matrix(y_test, y_pred)
    # df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    # show_confusion_matrix(df_cm)
    # test_acc, _ = eval_model(model, test_data_loader, loss_fn, device, len(data))
    # print(test_acc.item)