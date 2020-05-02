import torch
import torch.nn as nn
import torch.nn.functional as F


class HighWay(nn.Module):
        def __init__(self,h_size,word_dim,char_dim):
                super(HighWay,self).__init__()
                
                self.Dense1=nn.Linear(word_dim+char_dim,h_size)
                self.transform=nn.Linear(word_dim+char_dim,h_size)
            
        def forward(self,word,char):
                input=torch.cat([char,word],dim=1)
                
                output=F.relu_(self.Dense1(input))
                transform=torch.sigmoid_(self.transform(input))
                
                return output*transform+(1-transform)*input
  
          
class Convolution(nn.Module):
        def __init__(self,kernel_size,char_size,embedding_dim,output_channels):
                super(Convolution,self)    
            
                self.embedding=nn.Embedding(char_size,embedding_dim)
                
                self.conv1=nn.Conv1d(embedding_dim,output_channels,kernel_size[0])
                self.conv2=nn.Conv1d(embedding_dim,output_channels,kernel_size[1])
                self.conv3=nn.Conv1d(embedding_dim,output_channels,kernel_size[2])
                
        def forward(self,input):
                input=self.embedding(input)
                input=input.transpose(1,2)
                
                output1=self.conv1(input)
                output2=self.conv2(input)
                output3=self.conv3(input)
                
                output1=torch.max(output1,dim=2)
                output2=torch.max(output2,dim=2)
                output3=torch.max(output3,dim=2)
                
                output=torch.cat([output1,output2,output3],dim=1)
                
                return output
                
            
class OutputLayer(nn.Module):
        def __init__(self,hidden_size):
                super(OutputLayer,self)
                
                self.start=nn.Linear(hidden_size*10,1)
                self.end=nn.Linear(hidden_size*10,1)
                
                self.m_to_m2=nn.LSTM(hidden_size,hidden_size,num_layers=2,bidirectional=True)
                
        def forward(self,G,M):
                input1=torch.cat([G,M],dim=2)
                input2,(hidden,cell_state)=self.m_to_m2(M)
                
                starts=self.start(input1.transpose(0,1))
                ends=self.end(input2.transpose(0,1))
                
                return starts,ends
   
         
class ModelingLayer(nn.Module):
        def __init__(self,type,h_size):
                super(ModelingLayer,self)
                
                self.type=type
                self.m=nn.LSTM(h_size*8,h_size,bidirectional=True,num_layers=2)
                
        def forward(self,U_toggler,H_toggler,H):
                if self.type=='concat':
                        G=torch.cat([H,U_toggler,H*U_toggler,H*H_toggler])
                        M,(hidden,cell_state)=self.m(G)
                        
                        return G,M
                else:
                        raise NotImplementedError("Correct type has not been passed")
                        
                        
class ContextEmbedding(nn.Module):
        def __init__(self,hidden_size):
                super(ContextEmbedding,self)
                
                self.context_embedding=nn.LSTM(hidden_size,hidden_size,num_layers=2,bidirectional=True)
                
        def forward(self,input):
            
                output,(hidden_state,cell_state)=self.context_embedding(input)
                
                return output