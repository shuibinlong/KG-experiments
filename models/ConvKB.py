import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel

class ConvKB(BaseModel):
    def __init__(self, config):
        super(ConvKB, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(self.relation_cnt, self.emb_dim, padding_idx=0)
        self.input_drop = torch.nn.Dropout(kwargs.get('emb_dropout'))
        self.feature_map_drop = torch.nn.Dropout2d(kwargs.get('feature_map_dropout'))
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        self.conv1 = torch.nn.Conv2d(1, self.conv_out_channels, self.kernel_size, 1, 0, bias=kwargs.get('use_bias'))
        self.bn0 = torch.nn.BatchNorm2d(1)  # batch normalization over a 4D input
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels)
        filtered_h = (3 - self.kernel_size[0]) // self.stride + 1
        filtered_w = (self.emb_dim - self.kernel_size[1]) // self.stride + 1
        fc_length = self.conv_out_channels * filtered_h * filtered_w
        self.fc = torch.nn.Linear(fc_length, 1, bias=False)
        self.loss = ConvKBLoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    def forward(self, batch_h, batch_r, batch_t, batch_y=None):
        batch_size = batch_h.size(0)
        h = self.E(batch_h).unsqueeze(1)
        r = self.R(batch_r).unsqueeze(1)
        t = self.E(batch_t).unsqueeze(1)

        stacked_inputs = torch.cat([h, r, t], 1).unsqueeze(1) # (batch, 1, 3, dim)
        stacked_inputs = self.bn0(stacked_inputs)

        x = self.input_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        y = -x.view(-1)
        return self.loss(y, batch_y), y

class ConvKBLoss(BaseModel):
    def __init__(self):
        super().__init__()
    
    def forward(self, predict, label=None):
        loss = None
        if label is not None:
            y = F.softplus(predict * label)
            loss = torch.mean(y)
        return loss
