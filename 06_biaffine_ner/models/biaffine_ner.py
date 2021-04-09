import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from transformers import BertModel, BertPreTrainedModel


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Biaffine_NER(BertPreTrainedModel):

    def __init__(self, config):
        super(Biaffine_NER, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.start_activation = nn.LeakyReLU(0.1)
        self.end_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.end_activation = nn.LeakyReLU(0.1)
        self.biaffine = Biaffine(config.hidden_size, config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        outputs = self.dropout(outputs)
        start_info = self.start_activation(self.start_fc(outputs))
        end_info = self.end_activation(self.end_fc(outputs))

        logits = self.biaffine(start_info, end_info)

        input_mask = torch.unsqueeze(attention_mask, 2)
        input_mask_expand = torch.bmm(input_mask, torch.transpose(input_mask, 1, 2))  # [b_z, s_l, s_l]

        batch_size, seq_len = input_ids.size()
        tril_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), 0).expand(batch_size, seq_len, seq_len)
        label_mask = input_mask_expand * tril_mask

        return logits, label_mask