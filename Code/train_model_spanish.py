
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
# import argparse
import torchmetrics
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

import os

input_column = 'tweet'
output_column = ['class_encoded']
from torch.optim.lr_scheduler import ReduceLROnPlateau
# parser = argparse.ArgumentParser()
# parser.add_argument('-c', action='store_true')
# parser.add_argument('-e', '--excel', default='fully_processed.xlsx', type=str)
# parser.add_argument('-n', '--name', default='DenseNet',type=str)
# parser.add_argument('--dry', action='store_false')
# args = parser.parse_args()

######### all paths for the project
OR_PATH = os.getcwd()
os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'Dataset' + os.path.sep
MODEL_DIR = os.getcwd() + os.path.sep + 'Model' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

#dataset files
train_file = 'final_train_data_esp.csv'
test_file = 'final_test_data_esp.csv'
TRAIN_DATA_FILE = DATA_DIR +train_file
TEST_DATA_FILE = DATA_DIR +test_file

##model files
# LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.pkl'

input_column = 'tweet'
output_column = ['class_encoded']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)
epochs = 5
NUM_CLASSES = 10
MODEL_NAME_NLP ="bert-base-multilingual-cased"
BATCH_SIZE = 32 # --
CONTINUE_TRAINING = False
#CONTINUE_TRAINING = args.c
# MODEL_NAME = 'DenseNet' # --
MODEL_NAME = 'multilingua_spanish'

#MODEL_NAME = args.name
SAVE_MODEL = True # --
# SAVE_MODEL = args.dry
N_EPOCHS = 5 # --
LR = 0.01 # --
MOMENTUM = 0.9 # --
ES_PATIENCE = 5 # --
LR_PATIENCE = 1 # --
MAX_LENGTH = 128


'''
usage :
load the files and start running 
LABEL_ENCODER_FILE = MODEL_DIR + 'label_encoder.pkl'
TRAIN_DATA_LOADER =  MODEL_DIR + 'train_loader.pkl'
VAL_DATA_LOADER =  MODEL_DIR + 'val_loader.pkl'

with open(TRAIN_DATA_LOADER, 'rb') as f:
    loaded_train_loader = pickle.load(f)
with open(VAL_DATA_LOADER, 'rb') as f:
    loaded_val_loader = pickle.load(f)

for batch_X,batch_y in loaded_train_loader:
    print(batch_X)
    break
'''
def model_definition():

    #model = BertModel.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME_NLP, num_labels=10)  # 10 classes

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True)

    print(model, file=open(f'summary_{MODEL_NAME}.txt', 'w'))

    return model, optimizer, criterion, scheduler

def train_test(train_gen, test_gen, metrics_lst, metric_names, save_on, early_stop_patience):

    save_on = metric_names.index(save_on)

    model, optimizer, criterion, scheduler = model_definition()
    sig = nn.Sigmoid()

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

    if CONTINUE_TRAINING:
        model.load_state_dict(torch.load(f'model_{MODEL_NAME}.pt', map_location=device))
        model = model.to(device)
        print(f'Continuing Training - {MODEL_NAME}')
        model_save_epoch.append(0)
        model.eval()

        test_loss = 0
        steps_test = 0

        test_target_hist = list([])
        test_pred_labels = np.zeros(1)

        with tqdm(total=len(test_gen), desc=f'Saved Model') as pbar:
            with torch.no_grad():
                for xdata, xtarget in test_gen:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    optimizer.zero_grad()
                    output = model(xdata)
                    loss = criterion(output, xtarget)

                    steps_test += 1
                    test_loss += loss.item()

                    output_arr = output.detach().cpu().numpy()

                    if len(test_target_hist) == 0:
                        test_target_hist = xtarget.cpu().numpy()
                    else:
                        test_target_hist = np.vstack([test_target_hist, xtarget.cpu().numpy()])

                    pred_logit = output.detach().cpu()
                    pred_label = torch.where(pred_logit > 0.5, 1, 0)

                    test_pred_labels = np.vstack([test_pred_labels, pred_label.numpy()])

                    metrics_ = [metric(pred_label, xtarget.cpu()) for metric in metrics_lst]

                    pbar.update(1)
                    avg_test_loss = test_loss / steps_test
                    pbar.set_postfix_str(f'Test  Loss: {avg_test_loss:.5f}')

            test_loss_hist.append(avg_test_loss)
            test_metrics = [metric.compute() for metric in metrics_lst]
            test_metrics_hist.append(test_metrics)
            _ = [metric.reset() for metric in metrics_lst]

            met_test = test_metrics[save_on]
            met_test_best = met_test

        xstrres = 'Saved Model:'
        for name, value in zip(metric_names, test_metrics):
            xstrres = f'{xstrres} Test {name} {value:.5f}'
        print(xstrres)

    for epoch in range(N_EPOCHS):
        train_loss = 0
        steps_train = 0

        train_target_hist = list([])


        # --Start Model Training--
        model.train()

        with tqdm(total=len(train_gen), desc=f'Epoch {epoch}') as pbar:
            # for xdata, xtarget in train_gen:
            #     xdata, xtarget = xdata.to(device), xtarget.to(device)
            for step, batch in enumerate(train_gen):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                optimizer.zero_grad()
                logits = model(b_input_ids, b_input_mask).logits
                loss = criterion(logits, b_labels)
                loss.backward()
                optimizer.step()

                steps_train += 1
                train_loss += loss.item()
                train_loss_item.append([epoch, loss.item()])

                #output_arr = nn.functional.softmax(logits.detach(), dim=-1).cpu().numpy()  # Softmax for multi-class
                output_arr = nn.functional.softmax(logits.detach(), dim=-1).cpu().numpy()
                train_target = b_labels.cpu().numpy()
                if len(output_arr_hist) == 0:
                    output_arr_hist = output_arr
                else:
                    output_arr_hist = np.vstack([output_arr_hist, output_arr])
                if len(train_target_hist) == 0:
                    train_target_hist = train_target
                else:
                    train_target_hist = np.vstack([train_target_hist, train_target])
                pred_label = np.argmax(output_arr, axis=1)
                pred_one_hot = np.eye(NUM_CLASSES)[pred_label]

                pred_one_hot = torch.from_numpy(pred_one_hot)
                train_target = torch.from_numpy(train_target)
                metrics_ = [metric(pred_one_hot, train_target) for metric in metrics_lst]
                pbar.update(1)
                avg_train_loss = train_loss / steps_train
                pbar.set_postfix_str(f'Train Loss: {avg_train_loss:.5f}')

            train_loss_hist.append(avg_train_loss)
            train_metrics = [metric.compute() for metric in metrics_lst]
            train_metrics_hist.append([metric.compute() for metric in metrics_lst])
            _ = [metric.reset() for metric in metrics_lst]
        # --End Model Training--

        # --Start Model Test--
        test_loss = 0
        steps_test = 0
        test_target_hist = list([])
        test_pred_labels = np.zeros(NUM_CLASSES)
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
                    test_loss_item.append([epoch, loss.item()])

                    output_arr = nn.functional.softmax(logits.detach(), dim=-1).cpu().numpy()
                    test_target = b_labels.cpu().numpy()
                    if len(test_target_hist) == 0:
                        test_target_hist = output_arr
                    else:
                        test_target_hist = np.vstack([test_target_hist, output_arr])
                    pred_label = np.argmax(output_arr, axis=1)
                    pred_one_hot = np.eye(NUM_CLASSES)[pred_label]
                    pred_one_hot = torch.from_numpy(pred_one_hot)
                    test_target = torch.from_numpy(test_target)
                    test_pred_labels = np.vstack([test_pred_labels, pred_one_hot])

                    metrics_ = [metric(pred_one_hot, test_target) for metric in metrics_lst]

                    pbar.update(1)
                    avg_test_loss = test_loss / steps_test
                    pbar.set_postfix_str(f'Test  Loss: {avg_test_loss:.5f}')

            test_loss_hist.append(avg_test_loss)
            test_metrics = [metric.compute() for metric in metrics_lst]
            test_metrics_hist.append(test_metrics)
            _ = [metric.reset() for metric in metrics_lst]

            met_test = test_metrics[save_on]

        xstrres = f'Epoch {epoch}'
        for name, value in zip(metric_names, train_metrics):
            xstrres = f'{xstrres} Train {name} {value:.5f}'

        xstrres = xstrres + ' - '
        for name, value in zip(metric_names, test_metrics):
            xstrres = f'{xstrres} Test {name} {value:.5f}'

        print(xstrres)

        # Save Best Model
        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), f'model_{MODEL_NAME}.pt')

            xdf_dset_results = val_data.copy()
            # global var # might change this variable when test and validation are created
            test_pred_labels = test_pred_labels.astype(int)
            xdf_dset_results['results'] = [list(row) for row in test_pred_labels[1:]]

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
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        input_text = self.dataframe.loc[idx, input_column]
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
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = CustomDataset(val_data,tokenizer= self.tokenizer_)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
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
    train_data = full_df[full_df['split'] == 'train'].reset_index()
    val_data = full_df[full_df['split'] == 'val'].reset_index()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_NLP)
    data_loader = CustomDataLoader(tokenizer=tokenizer)
    train_loader, val_loader = data_loader.prepare_train_val_loader(train_data, val_data)

    metric_lst = [torchmetrics.Accuracy(task='binary'),
                  torchmetrics.Precision(task='binary'),
                  torchmetrics.Recall(task='binary'),
                  torchmetrics.AUROC(task='binary'),
                  torchmetrics.F1Score(task='binary')]
    metric_names = ['Accuracy',
                    'Precision',
                    'Recall',
                    'AUROC',
                    'F1Score']

    early_stop_patience = ES_PATIENCE
    save_on = 'F1Score'
    train_test(train_loader, val_loader, metric_lst, metric_names, save_on, early_stop_patience)

