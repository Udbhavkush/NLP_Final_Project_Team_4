
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
# import argparse
import torchmetrics
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

import os
import string
import re
input_column = 'tweet'
output_column = ['class_encoded']
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import SnowballStemmer
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score,  matthews_corrcoef

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', action='store_true')
# parser.add_argument('-e', '--excel', default='fully_processed.xlsx', type=str)
# parser.add_argument('-n', '--name', default='DenseNet',type=str)
# parser.add_argument('--dry', action='store_false')
# args = parser.parse_args()
train_file = 'final_train_data_esp.csv'
MODEL_NAME = 'distilbert-base-uncased-spanish-tweets-clf_20epoch'
#MODEL_NAME_NLP ="bert-base-multilingual-cased"
MODEL_NAME_NLP = "francisco-perez-sorrosal/distilbert-base-uncased-finetuned-with-spanish-tweets-clf"
#MODEL_NAME_NLP = "dccuchile/distilbert-base-spanish-uncased-finetuned-xnli"
#MODEL_NAME_NLP = "dccuchile/bert-base-spanish-wwm-cased-finetuned-mldoc"
######### all paths for the project
OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Dataset' + os.path.sep
MODEL_DIR = os.getcwd() + os.path.sep + 'Model' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

#dataset files

test_file = 'final_test_data_esp.csv'
TRAIN_DATA_FILE = DATA_DIR +train_file
TEST_DATA_FILE = DATA_DIR +test_file


##model files
# LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.pkl'

input_column = 'tweet'
output_column = ['class_encoded']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)
epochs = 30
NUM_CLASSES = 10

BATCH_SIZE = 180 # --
CONTINUE_TRAINING = False
#CONTINUE_TRAINING = args.c
# MODEL_NAME = 'DenseNet' # --


#MODEL_NAME = args.name
SAVE_MODEL = True # --
# SAVE_MODEL = args.dry
N_EPOCHS = 30 # --
LR = 0.01 # --
MOMENTUM = 0.9 # --
ES_PATIENCE = 5 # --
LR_PATIENCE = 1 # --
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

    xcont = 1
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

    #model = BertModel.from_pretrained(model_name)
    #model = BertForSequenceClassification.from_pretrained(MODEL_NAME_NLP, num_labels=10)  # 10 classes
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_NLP,num_labels =  NUM_CLASSES, ignore_mismatched_sizes=True)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True)

    print(model, file=open(f'summary_{MODEL_NAME}.txt', 'w'))

    return model, optimizer, criterion, scheduler

def train_test(train_gen, test_gen, metrics_lst, metric_names, save_on, early_stop_patience):

    #save_on = metric_names.index(save_on)

    model, optimizer, criterion, scheduler = model_definition()
    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    train_loss_hist = list([])
    test_loss_hist = list([])

    output_arr_hist = list([])
    pred_label_hist = list([])

    train_metrics_hist = list([])
    test_metrics_hist = list([])

    met_test_best = 0
    model_save_epoch = []
    pred_labels_per_hist = list([])
    pred_labels_per_hist_test = list([])
    for epoch in range(N_EPOCHS):
        train_loss = 0
        steps_train = 0
        pred_logits, real_labels = np.zeros((1, NUM_CLASSES)), np.zeros((1, NUM_CLASSES))
        train_target_hist = list([])
        train_hist = list([])
        test_hist = list([])
        # --Start Model Training--
        model.train()

        with tqdm(total=len(train_gen), desc=f'Epoch {epoch}') as pbar:
            for step, batch in enumerate(train_gen):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()
                logits = model(b_input_ids, b_input_mask).logits
                loss = criterion(logits, b_labels)
                loss.backward()
                optimizer.step()

                cont += 1
                steps_train += 1
                train_loss += loss.item()
                train_loss_item.append([epoch, loss.item()])
                pred_labels_per = logits.detach().to(torch.device('cpu')).numpy()
                train_target = b_labels.cpu().numpy()
                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])
                if len(train_hist) == 0:
                    train_hist = train_target
                else:
                    train_hist = np.vstack([train_hist, train_target])

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))

                pred_labels_per = nn.functional.softmax(logits.detach(), dim=-1).cpu().numpy()
                pred_label = np.argmax(pred_labels_per, axis=1)
                pred_one_hot = np.eye(NUM_CLASSES)[pred_label]

                pred_logits = np.vstack((pred_logits, pred_one_hot))
                real_labels = np.vstack((real_labels, train_target))

        pred_labels = pred_logits[1:]
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
        avg_train_loss = train_loss / steps_train

        #
        # train_loss_hist.append(avg_train_loss)
        # train_metrics = [metric.compute() for metric in metrics_lst]
        #     train_metrics_hist.append([metric.compute() for metric in metrics_lst])
        #     _ = [metric.reset() for metric in metrics_lst]
        # --End Model Training--

        # --Start Model Test--
        test_loss = 0
        steps_test = 0
        test_target_hist = list([])
        test_pred_labels = np.zeros(NUM_CLASSES)
        pred_test_logits, real_test_labels = np.zeros((1, NUM_CLASSES)), np.zeros((1, NUM_CLASSES))
        #pred_labels_per_hist =
        met_test = 0
        model.eval()
        with tqdm(total=len(test_gen), desc=f'Epoch {epoch}') as pbar:
            with torch.no_grad():
                for step, batch in enumerate(test_gen):
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch
                    optimizer.zero_grad()
                    logits = model(b_input_ids, b_input_mask).logits
                    loss = criterion(logits, b_labels)
                    steps_test += 1
                    test_loss += loss.item()
                    cont += 1
                    test_loss_item.append([epoch, loss.item()])
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

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)
        print(xstrres)
        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat
        print(xstrres)

        # Save Best Model
        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), f'model_{MODEL_NAME}.pt')

            xdf_dset_results = val_data.copy()
            # global var # might change this variable when test and validation are created
            test_pred_labels = pred_labels_test.astype(int)
            test_pred_labels_encoded = np.argmax(test_pred_labels, axis=1)
            xdf_dset_results['results'] = test_pred_labels_encoded #[list(row) for row in test_pred_labels[1:]]

            xdf_dset_results.to_excel(f'results_{MODEL_NAME}.xlsx', index=False)
            print('Model Saved !!')
            met_test_best = met_test
            model_save_epoch.append(epoch)

        # Early Stopping
        if epoch - model_save_epoch[-1] > early_stop_patience:
            print('Early Stopping !! ')
            break


class CustomDataset(Dataset):
    def __init__(self, dataframe,  max_length = MAX_LENGTH, tokenizer = None):
        self.dataframe = dataframe
        self.tokenizer_ = tokenizer
        self.max_length = max_length
        #self.stop_words = set(stopwords.words('spanish'))
       # self.stemmer = SnowballStemmer('spanish')
    def __len__(self):
        return len(self.dataframe)

    def normalize_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return text

    def normalize_spanish_text(self,text):
        #from pattern.es import parse, split
        #text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\w\s]', '', text)
        #tokens = word_tokenize(text)
        #tokens = [self.stemmer.stem(word) for word in tokens]
        #return ' '.join(tokens)
        return text
    def __getitem__(self, idx):
        input_text = self.dataframe.loc[idx, input_column]
        #input_text = self.normalize_spanish_text(input_text)
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
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=10)
        val_dataset = CustomDataset(val_data,tokenizer= self.tokenizer_)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=10)
        return train_loader, val_loader
    def prepare_test_dev_loader(self, test_data, dev_data):
        test_dataset = CustomDataset(test_data, self.tokenizer_, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        dev_dataset = CustomDataset(dev_data, self.tokenizer_, self.max_length)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader,dev_loader


if __name__ == '__main__':
    #full_df = pd.read_csv(TRAIN_DATA_FILE, engine='python', encoding='utf-8')
    full_df = pd.read_csv(TRAIN_DATA_FILE)
    #loaded_label_encoder = joblib.load(LABEL_ENCODER_FILE)
    #full_df = full_df[full_df['class'] != 'control']
    train_data = full_df[full_df['split'] == 'train'].reset_index()
    val_data = full_df[full_df['split'] == 'val'].reset_index()

    #tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_NLP)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_NLP)
    data_loader = CustomDataLoader(tokenizer=tokenizer)
    train_loader, val_loader = data_loader.prepare_train_val_loader(train_data, val_data)
    early_stop_patience = ES_PATIENCE
    save_on = 'F1Score'

    list_of_metrics = ['acc','f1_macro']
    list_of_agg = ['avg']
    train_test(train_loader, val_loader, list_of_metrics, list_of_agg,  save_on='f1_macro', early_stop_patience=ES_PATIENCE)


    #metrics_func(list_of_metrics, list_of_agg)
