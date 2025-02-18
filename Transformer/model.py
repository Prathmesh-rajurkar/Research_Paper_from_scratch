import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self,d_model,seq_len,dropout):
        super().__init__()
        self.d_model=d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # create a positional encoding
        pe = torch.zeros(seq_len,d_model)

        position  = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)
    
    def forward(self,x):
        x = x + (self.pe[:,:x.shape(1),:]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    
    def __init__(self,d_model,eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

class FeedForward(nn.Module):

    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h,dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model) 
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask=None,dropout=None):
        d_k = query.shape[-1]

        attention_scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0,-1e9)
        attention_scores = torch.softmax(attention_scores,dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores,value), attention_scores

    def forward(self,q,k,v,mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0],-1,self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],-1,self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],-1,self.h,self.d_k).transpose(1,2)

        x , self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.d_model)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self,d_model,dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x))) 

class EncoderLayer(nn.Module):
    
    def __init__(self,d_model,h,d_ff,dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout

        self.multi_head_attention = MultiHeadAttention(d_model,h,dropout)
        self.feed_forward = FeedForward(d_model,d_ff,dropout)
        self.residual_connection = ResidualConnection(d_model,dropout)
        self.residual_connection_ff = ResidualConnection(d_model,dropout)

    def forward(self,x,mask):
        x = self.residual_connection(x,lambda x: self.multi_head_attention(x,x,x,mask))
        x = self.residual_connection_ff(x,lambda x: self.feed_forward(x))
        return x

class Encoder(nn.Module):

    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x,mask):
        for layer in self.encoder_layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self,self_attention_block,cross_attention_block,feed_forward_block,dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = ResidualConnection(d_model,dropout)

    def forward(self,x,enc_output,src_mask,tgt_mask):
        x = self.residual_connections(x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections(x,lambda x: self.cross_attention_block(x,enc_output,enc_output,src_mask))
        x = self.residual_connections(x,lambda x: self.feed_forward_block(x))
        return x

class Decoder(nn.Module):
    
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x,enc_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,enc_output,src_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.linear(x),dim=-1)

class Transformer(nn.Module):

    def __init__(self,encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size,tgt_vocab_size,d_model,h,d_ff,N,dropout,src_seq_len,tgt_seq_len):
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)
 
    encoder_block = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForward(d_model,d_ff,dropout)
        encoder_block.append(EncoderLayer(d_model,h,d_ff,dropout))

    decoder_block = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForward(d_model,d_ff,dropout)
        decoder_block.append(DecoderLayer(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout))

    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

    