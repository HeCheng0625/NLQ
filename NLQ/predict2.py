import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertMultiClassifier, BertBiLSTM
from dataset import NLQDataset
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

def main():
    sentence = input("输入测试句子:")
    model = model = pickle.load(open("NLQ/model.pkl", 'rb'))
    model.eval()
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('chinese_wwm_ext_pytorch')
        token = tokenizer(sentence, add_special_tokens=True, padding='max_length', truncation=True, max_length=64)
        input_ids = torch.tensor(np.array(token['input_ids'])).reshape((1, 64)).to(device)
        token_type_ids = torch.tensor(np.array(token['token_type_ids'])).reshape((1, 64)).to(device)
        attention_mask = torch.tensor(np.array(token['attention_mask'])).reshape((1, 64)).to(device)
        out = model(input_ids, token_type_ids, attention_mask)
        out = out.cpu().numpy().tolist()
        # (B, L, class) -> (L, )
        y_pred = []
        for i in out:
            for j in i:
                y_pred.append(j.index(max(j)))
        y_pred = y_pred[1: len(sentence)+1]
        print("input sentecne:", sentence)
        resultDic = {0: '', 1: '', 2: '', 3:'', 4:'', 5: ''}
        attributeDic = {0: '教育经历', 1: '地区', 2: '年龄', 3: '性别', 4: '工作经历'}
        for i in range(len(y_pred)):
            resultDic[y_pred[i]] += sentence[i]
        for key, value in resultDic.items():
            if (key != 5):
                print(attributeDic[key]+':',value)

if __name__ == "__main__":
    main()