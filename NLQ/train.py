import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertMultiClassifier
from dataset import NLQDataset
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score, classification_report

BATCH_SIZE = 32
EPOCHS = 1
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
            loss = criterion(output, batch[3].float().to(device))
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
                y_pred = model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
                y_pred = y_pred.cpu().numpy().tolist()
                val_pred.extend(y_pred)
                val_true.extend(batch[3].cpu().numpy().tolist())
        for val in val_pred:
            for i in range(len(val)):
                if val[i] >= 0.5:
                    val[i] = 1
                else:
                    val[i] = 0
        print("\n Test Accuracy = {} \n".format(accuracy_score(val_pred, val_true)))
        
        print(classification_report(val_pred, val_true, digits=4))

def main():
    trainData = pickle.load(open("NLQ/trainData.pkl", 'rb'), encoding='utf-8')
    testData = pickle.load(open("NLQ/testData.pkl", 'rb'), encoding='utf-8')

    train_loader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testData, batch_size=BATCH_SIZE, shuffle=False)

    model = BertMultiClassifier().to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    train_and_eval(model, train_loader, test_loader, optimizer, criterion, device, EPOCHS)

    # with open('NLQ/model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

if __name__ == '__main__':
    main()