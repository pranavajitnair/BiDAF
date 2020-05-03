import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDirectionalAttention(nn.Module):
        def __init__(self,hidden_size):
                super(BiDirectionalAttention,self).__init__()
                
                self.q=nn.Linear(hidden_size*2,1)
                self.c=nn.Linear(hidden_size*2,1)
                self.qc=nn.Linear(hidden_size*2,1)
                
        def forward(self,H,U):
                q_len=U.shape[1]
                c_len=H.shape[1]
                
                l=[]
                for i in range(q_len):
                        temp_q=U[:,i].view(1,-1)
                        temp_qc=H*temp_q
                        
                        temp_qc=self.qc(temp_qc)
                        l.append(temp_qc)
                        
                qc=torch.cat(l,dim=2)
                q_temp=self.q(U).transpose(1,2).expand(-1,c_len,-1)
                c_temp=self.c(H).expand(-1,-1,q_len)
                
                s=qc+q_temp+c_temp
                
                q2c_atten=F.softmax(s,dim=2)

                U_toggler=torch.bmm(q2c_atten,U)
                
                b,_=torch.max(s,dim=2)
                c2q_atten=F.softmax(b,dim=1).unsqueeze(0)

                H_toggler=torch.bmm(c2q_atten,H)
                H_toggler= H_toggler.expand(-1,c_len,-1)
                
                return U_toggler,H_toggler