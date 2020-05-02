import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDirectionalAttention(nn.Module):
        def __init__(self,hidden_size):
                super(BiDirectionalAttention,self)
                
                self.q=nn.Linear(hidden_size*2,1)
                self.c=nn.Linear(hidden_size*2,1)
                self.qc=nn.Linear(hidden_size*2,1)
                
        def forward(self,H,U):
                q_len=U.shape[0]
                c_len=H.shape[0]
                
                l=[]
                for i in range(q_len):
                        temp_q=U[i].view(1,-1)
                        temp_qc=H*temp_q
                        
                        temp_qc=self.qc(temp_qc)
                        l.append(temp_qc)
                        
                qc=torch.cat(l,dim=1)
                q_temp=self.q(U).transpose(0,1).expand(c_len,-1)
                c_temp=self.c(H).expand(-1,q_len)
                
                s=qc+q_temp+c_temp
                
                q2c_atten=F.softmax(s,dim=0)
                U_toggler=torch.bmm(U,q2c_atten)
                
                
                b,_=torch.max(H,dim=1)
                c2q_atten=F.sotmax(b).view(1,-1)
                H_toggler=torch.bmm(c2q_atten,H).expand(c_len,-1)
                
                return U_toggler,H_toggler