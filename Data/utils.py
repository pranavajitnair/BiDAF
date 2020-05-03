import json
import string
import gensim.models as gs

import torch.nn.functional as F

def read_data(path):

        with open(path) as f:
                file=json.load(f)['data']
                
        return file    
            
def get_structure(file):
        char_set=set()
        char_set.add('̇')
        char_set.add('ꮉ')
        char_set.add('լ')
        char_set.add('ⲭ')
        char_set.add('պ')
        char_set.add('ј')
        char_set.add('є')
        char_set.add('ἥ')


        puncts=list(string.punctuation)
        output={'qids':[],'questions':[],'answers':[],'contexts':[]}
    
        for article in file:
                for paragraph in article['paragraphs']:
                    
                        x3=[x for x in paragraph['context'] if x not in puncts]
                        s=''
                        for char in x3:
                                s+=char
                                char_set.add(char)
                                
                        s=s.lower()
                        context=s.split()         
                        
                        l1=[]
                        l2=[]
                        l3=[]
                        
                        for qa in paragraph['qas']:
                                
                                s=''
                                s1=''                        
                                spaces=-1
                                
                                x2=[x for x in qa['answers'][0]['text'] if x not in puncts]              
                                for char in x2:
                                        s1+=char
                                        char_set.add(char)
                                        
                                s1=s1.lower()
                                answer=s1.split()
                                
                                x1=[x for x in qa['question'] if x not in puncts]
                                for char in x1:
                                        s+=char
                                        char_set.add(char)
                                        
                                s=s.lower()
                                
                                for i in range(len(context)):
                                        count=0
                                        for j in range(len(answer)):
                                                if i+j<len(context) and answer[j]==context[i+j]:
                                                        count+=1
                                                else:
                                                        break
                                        if count==len(answer):
                                                spaces=i
                                                break
                                            
                                if spaces!=-1:
                                        dict=[{'answer_start':spaces,'text':answer,'answer_end':spaces+len(answer)-1}]    
                                        l1.append(qa['id'])
                                        l2.append(s.split())
                                        l3.append(dict)
                        
                        if len(l1)!=0:
                                output['qids'].append(l1)
                                output['questions'].append(l2)
                                output['answers'].append(l3)
                                output['contexts'].append(context)               
                                
        return output,char_set
    
def get_index(output,char_set):
        final_output={'contexts':[],'questions':[]}
        l=list(char_set)
        char_to_int={}
        int_to_char={}
        
        char_to_int['PAD']=0
        int_to_char[0]='PAD'
        
        for i in range(len(l)):
                 int_to_char[i+1]=l[i]
                 char_to_int[l[i]]=i+1
                 
        for contexts in output['contexts']:
                context=[]
                ma=0
                
                for word in contexts:
                       ma=max(ma,len(word))
                       
                for word in contexts:
                        o=[]
                        
                        for character in word:
                                try:
                                    o.append(char_to_int[character])
                                except:
                                    aa=1

                        for _ in range(ma-len(o)):
                                o.append(char_to_int['PAD'])
                            
                        context.append(o)
                        
                final_output['contexts'].append(context)
                
        for question_set in output['questions']:
                sets=[]
                
                for question in question_set:
                        x=[]
                        ma=0
                        
                        for word in question:
                                ma=max(ma,len(word))
                                
                        for word in question:
                                o=[]
                                
                                for character in word:
                                        o.append(char_to_int[character])
                                for _ in range(ma-len(o)):
                                        o.append(char_to_int['PAD'])
                                        
                                x.append(o)
     
                        sets.append(x)
                        
                final_output['questions'].append(sets)
                
        return final_output,char_to_int,int_to_char,len(l)
    
def get_word_embeddings(input,hidden_size):
        embeds=gs.Word2Vec(input,min_count=1,size=hidden_size)
        
        return embeds
    
def get_question_set(input):
        l=[]
        
        for question_set in input['questions']:
                for question in question_set:
                        l.append(question)
                        
        return l
    
def get_prediction(start,end):
        start=F.softmax(start,dim=1)
        end=F.softmax(end,dim=1)
        
        start=start.squeeze(0)
        end=end.squeeze(0)
        
        ma=0
        start_pred=0
        end_pred=0
        
        for i in range(start.shape[0]):
                for j in range(i+1,end.shape[0]):
                        if int(start[i])*int(end[j])>ma:
                            
                                ma=int(start[i])*int(end[j])
                                start_pred=i
                                end_pred=j
                        
        return [start_pred,end_pred]
    
def get_f1(gold,pred):
        if gold[0]>pred[1] or gold[1]<pred[0]:
                return 0
        
        final=gold+pred
        final.sort()
        
        match=final[2]-final[1]+1
        precision=match/(pred[1]-pred[0]+1)
        recall=match/(gold[1]-gold[0]+1)
        
        f1=2*precision*recall(precision+recall)
        f1*=100
        
        return f1
    
def get_exact_match(gold,pred):
        
        return 100*(gold[0]==pred[0] and gold[1]==pred[1])