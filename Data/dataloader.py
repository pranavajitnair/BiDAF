import torch


class DataLoader(object):
        def __init__(self,output,char_output,word_dict):
                self.output=output
                self.char_output=char_output
                self.word_dict=word_dict
                
                self.counter=0
                self.counter1=0
                
        def get_next(self):
            
                l=[]
                for word in self.output['contexts'][self.counter]:
                        l.append(self.word_dict[word])      
                l=torch.tensor(l)
                
                l1=[]
                for word in self.char_output['contexts'][self.counter]:
                        l1.append(word)     
                l1=torch.tensor(l1)
                
                l2=[]
                for word in self.output['questions'][self.counter][self.counter1]:
                        l2.append(self.word_dict[word])
                l2=torch.tensor(l2)
                        
                l3=[]
                for word in self.char_output['questions'][self.counter][self.counter1]:
                        l3.append(word)
                l3=torch.tensor(l3)
                
                start=self.output['answers'][self.counter][self.counter1]['answer_start']
                end=self.output['answers'][self.counter][self.counter1]['answer_end']
                
                self.counter1+=1
                if self.counter1==len(self.output['questions'][self.counter]):
                        self.counter1=0
                        self.counter+=1
                        
                if self.counter==len(self.output['contexts']):
                        self.counter=0
                        self.counter1=0
                        
                return l,l1,l2,l3,start,end