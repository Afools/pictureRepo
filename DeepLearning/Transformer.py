import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PositionalEncodeing(nn.Module):
    def __init__(self,config):
        super().__init__()
        pos = torch.arange(0,config.max_position_embedding).unsqueeze(1)
        divterm = 1000**(-torch.arrange(0,config.hidden_size,2)/config.hidden_size)
        pe= torch.zeros(config.max_position_embedding,config.hidden_size)

        pe[:,0::2]=torch.sin(pos*divterm)
        pe[:,1::2]=torch.cos(pos*divterm)

        pe=pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    
    def forward (self, x:Tensor):
        return x+self.pe[:,:x.shape[1]]

class TransformerConfig():
    def __init__(self, 
                 hidden_size, 
                 head_num, 
                 head_size, 
                 max_embedding_len, 
                 hidden_dropout):
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = head_size
        self.max_embedding_len = max_embedding_len
        self.hidden_dropout=hidden_dropout
class MultiHeadAttention(nn.Module):
    def __init__(self, config:TransformerConfig):
        self.head_num=config.head_num
        self.head_size=config.head_size
        self.all_head_size=self.head_num*self.head_size

        self.query=nn.Linear(config.hidden_size,self.all_head_size)
        self.key=nn.Linear(config.hidden_size,self.all_head_size)
        self.value=nn.Linear(config.hidden_size,self.all_head_size)

        self.linear=nn.Linear(self.all_head_size,config.hidden_size)
        self.scaling=self.head_size**0.5

        self.dropout=nn.Dropout(config.hidden_dropout)

    def transpose_for_score(self, x):
        # [bz, len, h*d]->[bz,h,len,d]
        x.view((x.size()[0], x.size()[1], self.head_num, self.head_size))

        return x.permute(0,2,1,3)

    def merge_mask(self,key_padding_mask:Tensor, attn_mask:Tensor):
        # key padding mask :[bz, klen]->[bz, h, qlen, klen]
        if key_padding_mask is not None:
            bz, k_len=key_padding_mask.shape
            key_padding_mask=key_padding_mask.view((bz,1,1,k_len)).expand(
                -1,self.head_num,-1,-1
            )
            mask=key_padding_mask
        else:
            mask=None
        
        # attn_mask : [qlen, klen]->[bz,h,qlen,klen]
        if attn_mask is not None:
            dim=attn_mask.shape[0]
            attn_mask=attn_mask.view(1,1,dim,dim).expand(bz,self.head_num,-1,-1)
            if mask is not None:
                mask=mask.logical_or(attn_mask)
            else:
                mask=attn_mask
        return mask
    
    def attention_forward(self, q: Tensor, k:torch.Tensor, v:Tensor, mask, scaling,dropout):
        x = torch.matmul(q,k.transpose(-1,-2))*scaling
        if mask is not None:
            x.masked_fill(mask,float("-inf"))
        x = torch.softmax(x,dim=-1)
        x = F.dropout(x,p=dropout,training=self.training)
        x = torch.matmul(x,v)
        return x

    def forward(self, q, k, v, key_padding_mask, attn_mask):
        mask=self.merge_mask(key_padding_mask,attn_mask)
        q=self.transpose_for_score(self.query(q))
        k=self.transpose_for_score(self.key(k))
        v=self.transpose_for_score(self.value(v))

        score=self.attention_forward(q,k,v,mask,self.scaling,
                                     dropout=0.0 if not self.training else self.dropout)
        #[bz, h, len, d]
        score=score.transpose(1,2).contiguous()
        score=score.reshape(score.shape+(self.all_head_size))

        #[bz,len,h*d]
        score=self.linear(score)
        score=self.dropout(score)

        return score

