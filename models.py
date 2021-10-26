# models.py

import numpy as np
import collections
import torch.nn as nn
import torch
import random
import time
#####################
# MODELS FOR PART 1 #
#####################

class RNNOverWords(nn.Module):
    def __init__(self,dict_size,input_size,hidden_size,output_size,second_hidden_size,dropout=0,rnn_type='lstm'):
        super(RNNOverWords, self).__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding=nn.Embedding(dict_size,input_size)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.rnn_type=rnn_type
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers=1,dropout=dropout)
        self.first_linear=nn.Linear(hidden_size,second_hidden_size)
        self.non_linear=nn.Tanh()
        self.linear=nn.Linear(second_hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=2)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.xavier_uniform_(self.first_linear.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def convert_input(self,input,letter_indexer):
        return torch.LongTensor([letter_indexer.index_of(character) for character in input]).to(self.device)

    def forward(self, input):
        embedded_input=self.embedding(input)
        
        embedded_input=embedded_input.unsqueeze(1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float().to(self.device),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float().to(self.device))
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        output=self.softmax(self.linear(self.non_linear(self.first_linear(output))))
        hidden_state=self.softmax(self.linear(self.non_linear(self.first_linear(hidden_state))))
        return output, hidden_state, cell_state
    

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, model,vocab_index):
        self.model=model
        self.vocab_index=vocab_index
    def predict(self, context):
        input=self.model.convert_input(" "+context,self.vocab_index)
        output,log_probs,cells = self.model.forward(input)
        log_probs=torch.squeeze(log_probs)
        return torch.argmax(log_probs)

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    random.seed(0)
    start_time=time.time()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dictionary_size=len(vocab_index)
    #arbitrary input size cause this is the size of the embedding created
    input_size=12
    hidden_size=8
    second_hidden_size=4
    output_size=2
    model = RNNOverWords(dictionary_size,input_size,hidden_size, output_size, second_hidden_size)
    model.to(device)
    num_epochs=6

    training_data=[]
    for cons in train_cons_exs:
        training_data.append([" "+cons,0])
    for vowels in train_vowel_exs:
        training_data.append([" "+vowels,1])

    learning_rate=.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(0,num_epochs):
        total_loss=0
        random.shuffle(training_data)
        for i in range(0,len(training_data)):
            input=model.convert_input(training_data[i][0],vocab_index)
            y_onehot=torch.zeros(output_size).to(device)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(training_data[i][1],dtype=np.int64)).to(device), 1)
            model.zero_grad()
            output,log_probs,cells = model.forward(input)     
            log_probs=torch.squeeze(log_probs)
            loss = torch.neg(log_probs).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch}: {total_loss}')
    print(f'time: {time.time()-start_time}')
    return RNNClassifier(model,vocab_index)


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

class RNNLanguageModel(LanguageModel):
    def __init__(self,model,vocab_index):
        self.vocab_index=vocab_index
        self.model=model
    
    def get_next_char_log_probs(self,context):
        input=self.model.convert_input(f' {context}',self.vocab_index)
        output,hidden,cells=self.model.forward(input)
        hidden=torch.squeeze(hidden)
        return hidden.cpu().detach().numpy()
    
    def get_log_prob_sequence(self,next_chars,context):
        input=self.model.convert_input(f' {context}{next_chars}',self.vocab_index)
        output,hidden,cells=self.model.forward(input)
        output=torch.squeeze(output)
        #get the difference in output for the indexes and sum up the probabilities of every index 
        ret=0
        for i in range(0,len(next_chars)):
            index=i+len(context)
            character=self.vocab_index.index_of(next_chars[i])
            ret+=output[index][character].item()
        return ret


class RNNLanguageModelPredictions(nn.Module):
    def __init__(self,dict_size,input_size,hidden_size,output_size,dropout=0,rnn_type='lstm'):
        super(RNNLanguageModelPredictions, self).__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding=nn.Embedding(dict_size,input_size)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.rnn_type=rnn_type
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers=1,dropout=dropout)
        self.linear=nn.Linear(hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=2)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.xavier_uniform_(self.linear.weight)

    def convert_input(self,input,letter_indexer):
        return torch.LongTensor([letter_indexer.index_of(character) for character in input]).to(self.device)
    
    def forward(self, input):

        embedded_input=self.embedding(input)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float().to(self.device),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float().to(self.device))
        embedded_input=embedded_input.unsqueeze(1)
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        output=self.softmax(self.linear(output))
        hidden_state=self.softmax(self.linear(hidden_state))
        return output, hidden_state, cell_state

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dictionary_size=len(vocab_index)
    #arbitrary input size cause this is the size of the embedding created
    input_size=100
    hidden_size=50
    output_size=dictionary_size
    model = RNNLanguageModelPredictions(dictionary_size,input_size,hidden_size, output_size)
    model.to(device)

    learning_rate=.001

    criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    size_of_sentence=500

    training_data=np.empty((int((len(train_text)-1)/size_of_sentence),2),dtype=object)
    for i in range(0,len(train_text)-size_of_sentence,size_of_sentence):
        index=int(i/size_of_sentence)
        training_data[index][0]=train_text[i:i+size_of_sentence]
        training_data[index][1]=train_text[i:i+size_of_sentence+1]

    num_epochs=10
    for epoch in range(0,num_epochs): 
        #take the training text and divide it into 500 segments and train it on that 
        np.random.shuffle(training_data)
        total_loss=0
        for i in range(0,training_data.shape[0]):
            input=model.convert_input(f' {training_data[i][0]}',vocab_index)
            target=model.convert_input(training_data[i][1],vocab_index).to(device)
            model.zero_grad()
            output,hidden,cells=model.forward(input)
            output=torch.squeeze(output)
            loss=criterion(output,target.view(-1).long())
            total_loss+=loss
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch}: {total_loss}') 
    return RNNLanguageModel(model,vocab_index)

