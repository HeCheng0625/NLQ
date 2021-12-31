import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import pickle

class NLQDataset(Dataset):
    def __init__(self, filenameX, filenameY):
        self.tokenizer = BertTokenizer.from_pretrained('chinese_wwm_ext_pytorch')
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_ids = []
        self.load_data(filenameX, filenameY)
        
    def load_data(self, filenameX, filenameY):
        self.generateX = pickle.load(open(filenameX, 'rb'), encoding='utf-8')
        self.generateY = pickle.load(open(filenameY, 'rb'), encoding='utf-8')
        for i in tqdm(range(len(self.generateX))):
            token = self.tokenizer(self.generateX[i], add_special_tokens=True, padding='max_length', truncation=True, max_length=64)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_ids.append(np.array(self.generateY[i]))

    def raw_data(self, index):
        return self.generateX[index]

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_ids[index]

    def __len__(self):
        return len(self.input_ids)

trainData = NLQDataset('NLQ/trainX.pkl', 'NLQ/trainY.pkl')
testData = NLQDataset('NLQ/testX.pkl', 'NLQ/testY.pkl')

# some test code.
print('trainData size:', len(trainData))
input_ids, _, _, label_ids = trainData[100]
print('trainData[100]')
print(trainData.raw_data(100))
print(input_ids)
print(label_ids)

print('testData size:', len(testData))
input_ids, _, _, label_ids = testData[100]
print('testData[100]')
print(testData.raw_data(100))
print(input_ids)
print(label_ids)

with open('NLQ/trainData.pkl', 'wb') as f:
    pickle.dump(trainData, f)
with open('NLQ/testData.pkl', 'wb') as f:
    pickle.dump(testData, f)
