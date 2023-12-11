import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import os
import string
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score,  matthews_corrcoef, hamming_loss

# Getting the paths

MODEL_NAME = 'BERT_ENGLISH_ADAM'
MODEL_NAME_NLP = 'bert-base-uncased'
OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Dataset' + os.path.sep
MODEL_DIR = os.getcwd() + os.path.sep + 'Model' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

# Dataset files

train_file = 'final_train.csv'
test_file = 'final_test.csv'
TRAIN_DATA_FILE = DATA_DIR +train_file
TEST_DATA_FILE = DATA_DIR +test_file
input_column = 'tweet'
output_column = ['class_encoded']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)

# Define model parameters

epochs = 15
N_EPOCHS = 15
NUM_CLASSES = 10
BATCH_SIZE = 180
CONTINUE_TRAINING = False
SAVE_MODEL = True
LR = 0.01
MOMENTUM = 0.9
ES_PATIENCE = 5
LR_PATIENCE = 1
MAX_LENGTH = 128

def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        y_true_label_encoded = np.argmax(y_true, axis=1)
        y_pred_label_encoded = np.argmax(y_pred, axis=1)
        res = cohen_kappa_score(y_true_label_encoded, y_pred_label_encoded)
        #res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 0
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict

def model_definition():

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME_NLP, num_labels=10)
    model.load_state_dict(torch.load(f'model_BERT_ENGLISH_ADAM.pt', map_location=device))
    model = model.to(device)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay_rate': 0.1},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #      'weight_decay_rate': 0.0}
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters,
    #                   lr=2e-5,
    #                   eps=1e-8
    #                   )
    criterion = nn.CrossEntropyLoss()
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True)

    print(model, file=open(f'summary_{MODEL_NAME}.txt', 'w'))

    return model, criterion


def eval_model(test_gen, metrics_lst, metric_names):

    #save_on = metric_names.index(save_on)

    model, criterion = model_definition()
    cont = 0
    test_loss_item = list([])
    test_metrics_hist = list([])

        # --Start Model Test--
        test_loss = 0
        steps_test = 0
        test_target_hist = list([])
        test_pred_labels = np.zeros(NUM_CLASSES)
        pred_test_logits, real_test_labels = np.zeros((1, NUM_CLASSES)), np.zeros((1, NUM_CLASSES))
        #pred_labels_per_hist =
        met_test = 0
        model.eval()
        with tqdm(total=len(test_gen), desc=f'Evaluating') as pbar:
            with torch.no_grad():
                for step, batch in enumerate(test_gen):
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch
                    #optimizer.zero_grad()
                    logits = model(b_input_ids, b_input_mask).logits
                    loss = criterion(logits, b_labels)
                    steps_test += 1
                    test_loss += loss.item()
                    cont += 1
                    #test_loss_item.append([epoch, loss.item()])
                    pred_labels_per = logits.detach().to(torch.device('cpu')).numpy()
                    if len(pred_labels_per_hist_test) == 0:
                        pred_labels_per_hist_test = pred_labels_per
                    else:
                        pred_labels_per_hist_test = np.vstack([pred_labels_per_hist_test, pred_labels_per])
                    # output_arr = nn.functional.softmax(logits.detach(), dim=-1).cpu().numpy()
                    test_target = b_labels.cpu().numpy()
                    if len(test_target_hist) == 0:
                        test_target_hist = test_target
                    else:
                        test_target_hist = np.vstack([test_target_hist, test_target])

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    pred_labels_per = nn.functional.softmax(logits.detach(), dim=-1).cpu().numpy()
                    pred_label = np.argmax(pred_labels_per, axis=1)
                    pred_one_hot = np.eye(NUM_CLASSES)[pred_label]
                    pred_test_logits = np.vstack((pred_test_logits, pred_one_hot))
                    real_test_labels = np.vstack((real_test_labels, test_target))
            pred_labels_test = pred_test_logits[1:]
            test_metrics = metrics_func(list_of_metrics, list_of_agg, real_test_labels[1:], pred_labels_test)


        xstrres = "  "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
        print(xstrres)


class CustomDataset(Dataset):
    def __init__(self, dataframe,  max_length = MAX_LENGTH, tokenizer = None):
        self.dataframe = dataframe
        self.tokenizer_ = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def normalize_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return text

    def __getitem__(self, idx):
        input_text = self.dataframe.loc[idx, input_column]
        httpurl_pattern = r'HTTPURL'
        input_text = re.sub(httpurl_pattern, '', input_text)
        #input_text = self.normalize_text(input_text)
        label_cols = output_column
        labels = self.dataframe.loc[idx, label_cols].values[0]
        one_hot = np.eye(NUM_CLASSES)[labels]#.astype(int)
        label_tensor = torch.tensor(one_hot, dtype=torch.float32)
        if self.tokenizer_ == None:
            return input_text, label_tensor
        else:
            encoding = self.tokenizer_.encode_plus(
                input_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            return input_ids, attention_mask, label_tensor
class CustomDataLoader:
    def __init__(self,  batch_size=BATCH_SIZE, tokenizer = None, max_length = MAX_LENGTH):
        self.tokenizer_ = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
    def prepare_train_val_loader(self, train_data, val_data):
        train_dataset = CustomDataset(train_data,tokenizer=self.tokenizer_)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)
        val_dataset = CustomDataset(val_data,tokenizer= self.tokenizer_)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=8)
        return train_loader, val_loader
    def prepare_test_dev_loader(self, test_data, dev_data):
        test_dataset = CustomDataset(test_data, self.tokenizer_, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        dev_dataset = CustomDataset(dev_data, self.tokenizer_, self.max_length)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader,dev_loader


if __name__ == '__main__':
    full_df = pd.read_csv(TEST_DATA_FILE)
    #loaded_label_encoder = joblib.load(LABEL_ENCODER_FILE)
    #full_df = full_df[full_df['class'] != 'control']
    test_data = full_df[full_df['split'] == 'test'].reset_index()
    dev_data = full_df[full_df['split'] == 'dev'].reset_index()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_NLP)
    data_loader = CustomDataLoader(tokenizer=tokenizer)
    test_loader, dev_loader = data_loader.prepare_test_dev_loader(test_data, dev_data)

    list_of_metrics = ['acc','f1_macro']
    list_of_agg = ['avg']
    eval_model(test_loader, dev_loader, list_of_metrics, list_of_agg)



