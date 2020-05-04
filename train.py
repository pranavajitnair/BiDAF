import torch.nn as nn
import torch.optim as optim

import argparse

from Data.utils import read_data,get_structure,get_index,get_question_set,get_word_embeddings,get_prediction,get_f1,get_exact_match
from Data.dataloader import DataLoader

from Model.model import Model

def train(model,dataloader,lossFunction,optimizer,epochs,iterations,dataloader_validation,valid_iters):
        for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                loss=0
                
                for _ in range(iterations):
                        context_sentence,context_char,question_sentence,question_char,start,end=dataloader.get_next()
                        
                        start_logits,end_logits=model(context_sentence,question_sentence,question_char,context_char)
                        
                        loss+=lossFunction(start_logits,start)
                        loss+=lossFunction(end_logits,end)
                        
                final_loss=loss.item()
                
                loss.backward()
                optimizer.step()
                
                f1,em,valid_loss=validate(model,dataloader_validation,valid_iters,lossFunction)
                
                print('epoch=',epoch+1,'training loss=',final_loss/iterations,'validation loss=',valid_loss,'F1 score=',f1,'Exact Match score=',em)
 
                               
def validate(model,dataloader,iterations,lossFunction):
        model.eval()
        loss=0
        em=0
        f1=0
        
        for _ in range(iterations):
                context_sentence,context_char,question_sentence,question_char,start,end=dataloader.get_next()
                
                start_logits,end_logits=model(context_sentence,question_sentence,question_char,context_char)
                
                loss+=lossFunction(start_logits,start)
                loss+=lossFunction(end_logits,end)
                
                pred=get_prediction(start_logits,end_logits)
                gold=[int(start[0]),int(end[0])]
                
                f1+=get_f1(gold,pred)
                em+=get_exact_match(gold,pred)
                
        return f1/iterations,em/iterations,loss.item()/iterations

        
def main(args):
        path=args.train_path
        path_valid=args.dev_path
        
        data=read_data(path)
        data_valid=read_data(path_valid)
        
        hidden_size=args.hidden_size
        char_size=args.convolutions
        embedding_size=args.embedding_size
        char_embed_size=args.char_embedding_size
        
        dropout=args.dropout
        
        kernel_size=args.kernel_size1
        
        type=args.modeling_type
        
        output,char_set=get_structure(data)
        output_valid,char_set_valid=get_structure(data_valid)
        
        for char in char_set_valid:
                char_set.add(char)
        
        char_output,char_to_int,int_to_char,n_char=get_index(output,char_set)
        question_set=get_question_set(output)
        
        char_output_valid,x,y,z=get_index(output_valid,char_set)
        question_set_valid=get_question_set(output_valid)
        
        word_dict_question=get_word_embeddings(question_set,embedding_size)
        word_dict_context=get_word_embeddings(output['contexts'],embedding_size)
        
        word_dict_question_valid=get_word_embeddings(question_set_valid,embedding_size)
        word_dict_context_valid=get_word_embeddings(output_valid['contexts'],embedding_size)
        
        dataLoader=DataLoader(output,char_output,word_dict_context,word_dict_question)
        dataLoader_valid=DataLoader(output_valid,char_output_valid,word_dict_context_valid,word_dict_question_valid)
        
        model=Model(embedding_size,char_size,hidden_size,kernel_size,n_char,type,char_embed_size,dropout).cuda()
        
        epochs=args.epochs
        iterations=len(question_set)
        iterations_validation=len(question_set_valid)
        
        lossFunction=nn.CrossEntropyLoss()
        optimizer=optim.Adamax(model.parameters(),lr=args.learning_rate)
        
        train(model,dataLoader,lossFunction,optimizer,epochs,iterations,dataLoader_valid,iterations_validation)
        
        
def setup():
        parser=argparse.ArgumentParser('options for file')
        
        parser.add_argument('-- dev_path',type=str,default='/home/pranav/ml/data/SQuAD 1.1/dev-v1.1.json',help='enter development file path')
        parser.add_argument('--train_path',type=str,default='/home/pranav/ml/data/SQuAD 1.1/train-v1.1.json',help='enter training file path')
        
        parser.add_argument('--learning_rate',type=float,default=0.5,help='learning rate')
        parser.add_argument('--epochs',type=int,default=12)
        parser.add_argument('--modeling_type',type=str,default='concat',help='enter type for modeling')
        parser.add_argument('--hidden_size',type=int,default=100,help="hidden sizes for LSTM's")
        parser.add_argument('--convolutions',type=int,default=100,help='output channels for  Conv1D')
        parser.add_argument('--embedding_size',type=int,default=100,help='embedding size for Word2Vec')
        parser.add_argument('--kernel_size1',type=int,default=5,help='first kernel size')
        parser.add_argument('--dropout',type=float,default=0.2)
        parser.add_argument('--char_embedding_size',type=int,default=8)
        
        args=parser.parse_args()
        
        return args
        
if __name__=='__main__':
        args=setup()
        main(args)