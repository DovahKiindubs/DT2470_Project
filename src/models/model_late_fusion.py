"""Two-branch CRNN late fusion at the representation level, without fixed feature weights."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from models.model_crnn import downsample_lengths


class SequenceBranch(nn.Module):
    def __init__(self,input_channels,conv_channels=64,hidden_size=64,dropout=.2):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(input_channels,conv_channels,5,stride=2,padding=2),
            nn.BatchNorm1d(conv_channels),nn.GELU(),
            nn.Conv1d(conv_channels,conv_channels,5,stride=2,padding=2),
            nn.BatchNorm1d(conv_channels),nn.GELU(),nn.Dropout(dropout))
        self.gru=nn.GRU(conv_channels,hidden_size,batch_first=True,bidirectional=True)
        self.attention=nn.Linear(hidden_size*2,1)

    def forward(self,x,lengths,return_attention=False):
        encoded=self.conv(x).transpose(1,2)
        reduced=downsample_lengths(lengths).clamp(min=1,max=encoded.shape[1])
        packed=pack_padded_sequence(encoded,reduced.cpu(),batch_first=True,enforce_sorted=False)
        output,_=self.gru(packed)
        output,_=pad_packed_sequence(output,batch_first=True,total_length=encoded.shape[1])
        positions=torch.arange(output.shape[1],device=output.device)[None,:]
        mask=positions<reduced.to(output.device)[:,None]
        logits=self.attention(output).squeeze(-1).masked_fill(~mask,-torch.inf)
        weights=torch.softmax(logits,dim=1)
        pooled=torch.sum(output*weights.unsqueeze(-1),dim=1)
        return (pooled,weights,reduced) if return_attention else pooled


class LateFusionCRNN(nn.Module):
    """Independent native-time branches, equal-size normalized projections, learned MLP fusion."""
    def __init__(self,mode='late-fusion',cqt_conv_channels=64,cqt_hidden_size=64,
                 rhythm_conv_channels=32,rhythm_hidden_size=32,embedding_size=64,dropout=.2):
        super().__init__()
        if mode not in {'cqt-only','rhythm-only','late-fusion'}:
            raise ValueError(f'Unknown branch mode: {mode}')
        self.mode=mode
        self.cqt_branch=(SequenceBranch(88,cqt_conv_channels,cqt_hidden_size,dropout)
                         if mode in {'cqt-only','late-fusion'} else None)
        self.rhythm_branch=(SequenceBranch(9,rhythm_conv_channels,rhythm_hidden_size,dropout)
                            if mode in {'rhythm-only','late-fusion'} else None)
        self.cqt_projection=(nn.Sequential(nn.Linear(cqt_hidden_size*2,embedding_size),
                            nn.GELU(),nn.LayerNorm(embedding_size)) if self.cqt_branch else None)
        self.rhythm_projection=(nn.Sequential(nn.Linear(rhythm_hidden_size*2,embedding_size),
                               nn.GELU(),nn.LayerNorm(embedding_size)) if self.rhythm_branch else None)
        fusion_size=embedding_size*(2 if mode=='late-fusion' else 1)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(fusion_size,embedding_size),
                                nn.GELU(),nn.Dropout(dropout),nn.Linear(embedding_size,1))

    def forward(self,cqt,cqt_lengths,rhythm,rhythm_lengths,return_attention=False):
        vectors=[];details={}
        if self.cqt_branch:
            result=self.cqt_branch(cqt,cqt_lengths,return_attention)
            pooled,weights,reduced=result if return_attention else (result,None,None)
            vectors.append(self.cqt_projection(pooled));details['cqt']=(weights,reduced)
        if self.rhythm_branch:
            result=self.rhythm_branch(rhythm,rhythm_lengths,return_attention)
            pooled,weights,reduced=result if return_attention else (result,None,None)
            vectors.append(self.rhythm_projection(pooled));details['rhythm']=(weights,reduced)
        representation=torch.cat(vectors,dim=1)
        score=self.head(representation).squeeze(-1)
        return (score,representation,details) if return_attention else score
