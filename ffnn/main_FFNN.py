#dependencies

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import data as d_read
from torch.autograd import gradcheck
import torchwordemb

# Architecture feed forward NN presented by Bengio

## Word features, (|V|x m) matrix, where |V| = vocab size


# The yi are the unnormalized log-probabilities for each output word i, computed as follows, with
# parameters b,W,U,d and H:    y = b+Wx+Utanh(d+Hx)

## Let h be the number of hidden units and m the number of features for each word.


## Output biases b (with |V| elements), the hidden layer biases d (with
## h elements), the hidden-to-output weights U (a |V|x h matrix), the word features to output weights
## W (a |V| x (n−1)m matrix), the hidden layer weights H (a h x (n−1)m matrix), and the word
## features C (a |V| x m matrix).


# In the model below there is an optional argument to include  W : input to output neurons. 




# read in reduced data set

data = d_read.Corpus("/language-modeling-nlp1/data/penn")

train_data = data.train
valid_data = data.valid
test_data = data.test

# word embeddings



# preparing ngrams of size n = 6

ngrams = [([train_data[i],train_data[i+1],train_data[i+2],train_data[i+3],train_data[i+4]],train_data[i+5])
         for i in range(len(train_data) - 5)]

ngrams_valid = [([valid_data[i],valid_data[i+1],valid_data[i+2],valid_data[i+3],valid_data[i+4]],valid_data[i+5])
         for i in range(len(valid_data) - 5)]



## load embeddings

vocab, vec = torchwordemb.load_glove_text("language-modeling-nlp1/embeddings/glove.6b/glove.6B.50d.txt")

# vocab of treebank
vocab_tb = data.dictionary.word2idx.keys()

# mean  vec of all embeddings 
mean_vec = torch.mean(vec,0).view(1,50)


# mean vec for digits
numvec = vec[vocab["0"],:].view(1,50)
numvec = torch.cat((vec[vocab["1"],:].view(1,50),numvec),0)
numvec = torch.cat((vec[vocab["2"],:].view(1,50),numvec),0)
numvec = torch.cat((vec[vocab["3"],:].view(1,50),numvec),0)
numvec = torch.cat((vec[vocab["4"],:].view(1,50),numvec),0)
numvec = torch.cat((vec[vocab["5"],:].view(1,50),numvec),0)
numvec = torch.cat((vec[vocab["6"],:].view(1,50),numvec),0)
numvec = torch.cat((vec[vocab["7"],:].view(1,50),numvec),0)
numvec = torch.cat((vec[vocab["8"],:].view(1,50),numvec),0)
numvec = torch.cat((vec[vocab["9"],:].view(1,50),numvec),0)
mean_num = torch.mean(numvec,0)



count_uk = 0 # counts unknown words that do not contain "-"
count_num = 0 # count numerics
count_sp = 0 # counts words that contain "-"


embeddings = torch.randn((1,50))


for word in vocab_tb:

    if word not in vocab.keys():

        if word == "N":
            new = mean_num.view(1,50)
            embeddings = torch.cat((embeddings,new), 0)
            count_num+=1

        elif len(word.split("-")) > 1:
            count_sp +=1
            word = word.split(word)
            
            if any(x not in vocab.keys() for x in word):
                new = mean_vec
                embeddings = torch.cat((embeddings,new), 0)
            else:

                new = torch.zeros(50).view(1,50)
                for k in word:
                    embed = vec[vocab[k],:].view(1,50)
                    new+=embed
                new = new/len(word)
                embeddings = torch.cat((embeddings,new), 0)
        else:
            new = mean_vec.view(1,50)
            embeddings = torch.cat((embeddings,new), 0)
            count_uk +=1



    else:
        new = vec[vocab[word],:].view(1,50)
        embeddings = torch.cat((embeddings,new), 0)

#print(embeddings.size())


# exclude first row as it was just used for initialization of the tensor
embeddings = embeddings[1:][:]


# learning rate scheduler

def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


## from pytorch.org, here adapted to Bengio et al. 2003

# model 



class FFNN(nn.Module):
# added optional input_to_hidden connections

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size, input_to_output = False):
        super(FFNN, self).__init__()

        #embedding = nn.Embedding(embeddings.size(0), embeddings.size(1))
        #embedding.weight = nn.Parameter(embeddings)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = nn.Parameter(embeddings)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.input_to_output = input_to_output
        if input_to_output == True:
            self.linear3 = nn.Linear(context_size*embedding_dim,vocab_size)
            
        #biases for hidden and output are automatically included in nn.Linear
        
        

    def forward(self, inputs,input_to_output = False):
        
       
        embeds = self.embeddings(inputs).view((1, -1)) ## just creates a 1 dimensional array from the given arrays
        
        out = torch.nn.functional.tanh(self.linear1(embeds))
        
        if self.input_to_output == True:
            input2out = self.linear3(embeds)
            out = input2out + out
        else: 
            out =  self.linear2(out)
            
        
        
        return out


# evaluation function

def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0
    
    for words, target in data:
        lookup_tensor = Variable(torch.LongTensor([words]))
        scores = model(lookup_tensor)
        predict = scores.data.numpy().argmax(axis=1)[0]

        if predict == target:
            correct += 1

    return correct, len(data), correct/len(data)



CONTEXT_SIZE = 5   # 6 gram

EMBEDDING_DIM = 50

HIDDEN_SIZE = 60

ntokens = len(data.dictionary)

word2idx = data.dictionary.word2idx
idx2word = data.dictionary.idx2word
losses = []


model = FFNN(ntokens, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE)

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(3):
    print("start training epoch: {}".format(epoch))
    
    total_loss = torch.Tensor([0])

    mini_batch = 0 # counter used for mini batches

    # not sure about hyperparams
    optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7)

    for context, target in ngrams:

        mini_batch+=1

        
        context_var = torch.autograd.Variable(torch.LongTensor(context))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        probs = model(context_var)
        
        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = nn.CrossEntropyLoss()

        

        loss_out = loss(probs, torch.autograd.Variable(torch.LongTensor([target])))

        
        
       
        # calculates gradient
       
        loss_out.backward()

        if mini_batch == 32:

            # weight update after gradients of minibatch are collected
            optimizer.step()
            mini_batch = 0

        #b = list(model.parameters())[0].clone()
        #print(torch.equal(a.data, b.data))

 


        total_loss += loss_out.data

    losses.append(total_loss)
    
    corr, _, acc = evaluate(model, ngrams_valid)
    print("epoch {}: test acc={} % and correctly predicted = {}".format(epoch, round(100 *acc,5), corr))    
    print("loss this epoch: {}".format(list(total_loss)))


# I do not know which format to use to save

## http://pytorch.org/docs/master/notes/serialization.html
torch.save(model.state_dict(), "PATH")




