import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel

class TransE(BaseModel):
    def __init__(self, config):
        super(TransE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim')
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(self.relation_cnt, self.emb_dim, padding_idx=0)
        self.p_norm = kwargs.get('p_norm')
        self.loss = TransELoss(self.device, kwargs.get('margin'))
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)
    
    def forward(self, batch_h, batch_r, batch_t, batch_y=None):
        h = self.E(batch_h)
        r = self.R(batch_r)
        t = self.E(batch_t)

        x = torch.norm(h + r - t, self.p_norm, 1)
        return self.loss(x, batch_y), x
    
class TransELoss(BaseModel):
    def __init__(self, device, margin):
        super().__init__()
        self.device = device
        self.loss = torch.nn.MarginRankingLoss(margin, False)
    
    def forward(self, predict, label=None):
        loss = None
        if label is not None:
            pos = torch.where(label == 1)
            neg = torch.where(label == -1)
            pos_score = predict[pos]
            neg_score = predict[neg]
            y = torch.tensor([-1.0]).to(self.device)  # a positive triple should have a smaller score than a negative triple
            loss = self.loss(pos_score, neg_score, y)
        return loss