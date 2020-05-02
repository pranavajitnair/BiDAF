import torch.nn as nn

from layers import HighWay,Convolution,OutputLayer,ModelingLayer,ContextEmbedding
from attention import BiDirectionalAttention 


class Model(nn.Module):
        def __init__(self,embed_size,char_size,hidden_size,kernel_size,n_char,type):
                super(Model,self)
                
                self.highway=HighWay(hidden_size,embed_size,char_size)
                self.char_embedding=Convolution(kernel_size,n_char,embed_size,char_size)
                self.context_embedding=ContextEmbedding(hidden_size)
                
                self.attention=BiDirectionalAttention(hidden_size)
                
                self.modelinglayer=ModelingLayer(type,hidden_size)
                
                self.outputlayer=OutputLayer(hidden_size)
                
        def forward(self,sentence_context,sentence_question,char_sentence_question,char_sentence_context):
                char_embeds_context=self.char_embedding(char_sentence_context)
                input_context=self.highway(sentence_context,char_embeds_context)
                
                H=self.context_embedding(input_context)
                
                char_embeds_question=self.char_embedding(char_sentence_question)
                input_question=self.highway(sentence_question,char_embeds_question)
                
                U=self.context_embedding(input_question)
                
                U_toggler,H_toggler=self.attention(H,U)
                
                G,M=self.modelinglayer(U_toggler,H_toggler,H)
                
                start,end=self.outputlayer(G,M)
                
                return start,end