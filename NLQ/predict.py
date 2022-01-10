import torch
import numpy as np
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertMultiClassifier
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    tokenizer = BertTokenizer.from_pretrained('chinese_wwm_ext_pytorch')
    sentence = input()
    token = tokenizer(sentence, add_special_tokens=True, padding='max_length', truncation=True, max_length=64)
    input_ids = torch.tensor(np.array(token['input_ids'])).reshape((1, 64)).to(device)
    token_type_ids = torch.tensor(np.array(token['token_type_ids'])).reshape((1, 64)).to(device)
    attention_mask = torch.tensor(np.array(token['attention_mask'])).reshape((1, 64)).to(device)
    print(input_ids)
    print(token_type_ids)
    print(attention_mask)
    model = pickle.load(open("NLQ/model.pkl", 'rb'))
    model.eval()
    with torch.no_grad():
        print(model(input_ids, token_type_ids, attention_mask))


if __name__ == '__main__':
    main()