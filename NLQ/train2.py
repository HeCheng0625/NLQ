import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertMultiClassifier, BertBiLSTM
from dataset import NLQDataset
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score, classification_report

BATCH_SIZE = 32
EPOCHS = 3
learning_rate = 2e-5
weight_decay = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

def train_and_eval(model, train_loader, test_loader, optimizer, criterion, device, epoch):
    for i in range(epoch):
        model.train()
        print("***** Running training epoch {} *****".format(i+1))
        train_loss_sum = 0.0
        for batch_index, batch in tqdm(enumerate(train_loader)):
            output = model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
            # ner的输出为(batch_size, seq_length, tag_nums), target为(batch_size, seq_length)
            loss = criterion(output.permute(0,2,1), batch[4].long().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            if (batch_index + 1) % (len(train_loader)//100) == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f}".format(
                          i+1, batch_index+1, len(train_loader), train_loss_sum/(batch_index+1)))

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch_index, batch in tqdm(enumerate(test_loader)):
                out = model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
                out = out.cpu().numpy().tolist()
                # (B, L, class)
                y_pred = []
                for i in out:
                    for j in i:
                        y_pred.append(j.index(max(j)))
                val_pred.extend(y_pred)
                # (B, L)
                val_true.extend(batch[4].view(-1).cpu().numpy().tolist())
        # for val in val_pred:
        #     for i in range(len(val)):
        #         if val[i] >= 0.5:
        #             val[i] = 1
        #         else:
        #             val[i] = 0
        print("\n Test Accuracy = {} \n".format(accuracy_score(val_pred, val_true)))
        print("{0:'教育经历', 1:'地区', 2:'年龄', 3:'性别', 4:'工作经历', 5: 其余, 6: <CLS>(bert头), 7: <SEP>(bert尾) 8: <PAD>(bert填充)}")
        print(classification_report(val_pred, val_true, digits=4))

def main():
    trainData = pickle.load(open("NLQ/trainData.pkl", 'rb'), encoding='utf-8')
    testData = pickle.load(open("NLQ/testData.pkl", 'rb'), encoding='utf-8')

    train_loader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False)

    model = BertBiLSTM().to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_and_eval(model, train_loader, test_loader, optimizer, criterion, device, EPOCHS)

    # with open('NLQ/model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

if __name__ == '__main__':
    main()