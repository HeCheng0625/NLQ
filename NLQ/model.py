import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers import BertModel, BertConfig

class BertMultiClassifier(nn.Module):
    def __init__(self, bert_path = "chinese_wwm_ext_pytorch", out_size = 5):
        super(BertMultiClassifier, self).__init__()
        self.config = BertConfig(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.lin = nn.Linear(self.config.hidden_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        output = self.bert(input_ids, attention_mask, token_type_ids)
        output = output[1]
        output = self.lin(output)
        # (B, 5)
        output = self.sigmoid(output)
        return output

class BertBiLSTM(nn.Module):
    # {0:'教育经历', 1:'地区', 2:'年龄', 3:'性别', 4:'工作经历', 5: 其余, 
    # 6: <CLS>(101), 7: <SEP>(102) 8: <PAD>(0)}
    def __init__(self, bert_path = "chinese_wwm_ext_pytorch", out_size = 9):
        super(BertBiLSTM, self).__init__()
        self.config = BertConfig(bert_path)  
        self.bert = BertModel.from_pretrained(bert_path)
        self.bilstm = nn.LSTM(self.config.hidden_size, 128, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(256, out_size)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # (B, L)->(B, L, 768)
        emb = self.bert(input_ids, attention_mask, token_type_ids)
        emb = emb[0]
        # packed = pack_padded_sequence(emb, lengths, batch_first=True)
        # (B, L, 768)->(B, L, 256)
        rnn_out, _ = self.bilstm(emb)
        # (B, L, 256)
        # rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(rnn_out)
        return scores;

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)