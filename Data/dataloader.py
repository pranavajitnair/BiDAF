import torch


class DataLoader(object):
        def __init__(self,output,char_output,word_dict_context,word_dict_question):
                self.output=output
                self.char_output=char_output
                
                self.word_dict_context=word_dict_context
                self.word_dict_question=word_dict_question
                
                self.counter=0
                self.counter1=0
                
        def get_next(self):
            
                l=[]
                for word in self.output['contexts'][self.counter]:
                        l.append(self.word_dict_context[word])      
                l=torch.tensor(l).unsqueeze(0).cuda()
                
                l1=[]
                for word in self.char_output['contexts'][self.counter]:
                        l1.append(word)     
                l1=torch.tensor(l1,dtype=torch.long).cuda()
                
                l2=[]
                for word in self.output['questions'][self.counter][self.counter1]:
                        l2.append(self.word_dict_question[word])
                l2=torch.tensor(l2).unsqueeze(0).cuda()
                        
                l3=[]
                for word in self.char_output['questions'][self.counter][self.counter1]:
                        l3.append(word)
                l3=torch.tensor(l3,dtype=torch.long).cuda()
                
                start=self.output['answers'][self.counter][self.counter1][0]['answer_start']
                end=self.output['answers'][self.counter][self.counter1][0]['answer_end']
                
                start=torch.tensor(start).view(1).cuda()
                end=torch.tensor(end).view(1).cuda()
                
                self.counter1+=1
                if self.counter1==len(self.output['questions'][self.counter]):
                        self.counter1=0
                        self.counter+=1
                        
                if self.counter==len(self.output['contexts']):
                        self.counter=0
                        self.counter1=0
                        
                return l,l1,l2,l3,start,end