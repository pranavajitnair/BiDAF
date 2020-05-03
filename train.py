import torch.nn as nn
import torch.optim as optim

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

        
def main():
        path='/home/pranav/ml/data/SQuAD 1.1/train-v1.1.json'
        path_valid='/home/pranav/ml/data/SQuAD 1.1/dev-v1.1.json'
        
        data=read_data(path)
        data_valid=read_data(path_valid)
        
        hidden_size=100
        char_size=100
        embedding_size=100
        
        kernel_size=[2,2,2]
        
        type='concat'
        
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
        
        model=Model(embedding_size,char_size,hidden_size,kernel_size,n_char,type).cuda()
        
        epochs=1200
        iterations=len(question_set)
        iterations_validation=len(question_set_valid)
        
        lossFunction=nn.CrossEntropyLoss()
        optimizer=optim.Adamax(model.parameters(),lr=0.07)
        
        train(model,dataLoader,lossFunction,optimizer,epochs,iterations,dataLoader_valid,iterations_validation)
        
if __name__=='__main__':    
        main()