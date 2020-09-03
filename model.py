import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, pretrained_dir, tag_num, lr = 5e-5, device = 'cpu', fineTune = False):
        super().__init__()
        # Shared
        self.bert = BertModel.from_pretrained(pretrained_dir)        
        self.device = device
        self.fineTune = fineTune
        num_filters = 200
        kernel_sizes = [6, 8, 10]
        bert_hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.5)
        # NER
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=bert_hidden_size, hidden_size=bert_hidden_size//2, batch_first=True)
        self.fc = nn.Linear(bert_hidden_size, tag_num)
        # RC
        self.convs_rc = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, bert_hidden_size)) for k in kernel_sizes])
        self.linear = nn.Linear(len(kernel_sizes) * num_filters, 2)
        # Triage
        self.triageCls = nn.Linear(len(kernel_sizes) * num_filters, 2)
        # self.triageCls = nn.Linear(bert_hidden_size, 2)

        self.optimizer = optim.Adam([
            {'params': self.bert.parameters(), 'lr': 1e-5},
            {'params': self.lstm.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.linear.parameters()},
            {'params': self.triageCls.parameters()},
            {'params': self.convs_rc.parameters()},
            ], lr = lr
        )

        if device != 'cpu':
            self.cuda(device=self.device)

    def conv_and_pool(self, x, conv):
        x = conv(x.unsqueeze(1))
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forwardNER(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        # with torch.no_grad():
        last_hidden_state, _ = self.bert(x)
        last_hidden_state, _ = self.lstm(last_hidden_state)
        logits = self.fc(last_hidden_state)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

    def forwardRC(self, x, y):
        """ Take a mini-batch of Examples, compute the probability of relation
        @param examples (List[InputExample]): list of InputExample

        @returns  loss
        """
        y = y.to(self.device)
        for k in x:
            x[k] = x[k].to(self.device)
        if self.fineTune:
            last_hidden_state, pooled_output = self.bert(**x)
        else:
            with torch.no_grad():
                last_hidden_state, pooled_output = self.bert(**x)
        out = torch.cat([self.conv_and_pool(last_hidden_state, conv) for conv in self.convs_rc], 1)
        out = self.dropout(out)
        logits = self.linear(out)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

    def forwardTriage(self, x, y):
        """ Take a mini-batch of Examples, compute the probability of relation
        @param examples (List[InputExample]): list of InputExample

        @returns  loss
        """
        y = y.to(self.device)
        for k in x:
            x[k] = x[k].to(self.device)
        if self.fineTune:
            last_hidden_state, pooled_output = self.bert(**x)
        else:
            with torch.no_grad():
                last_hidden_state, pooled_output = self.bert(**x)
        out = torch.cat([self.conv_and_pool(last_hidden_state, conv) for conv in self.convs_rc], 1)
        out = self.dropout(out)
        logits = self.triageCls(out)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
