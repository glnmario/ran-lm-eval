
# word embeddings
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import data2 as d_read2
import data as d_read
from torch.autograd import gradcheck
import torchwordemb


vocab, vec = torchwordemb.load_glove_text("language-modeling-nlp1/embeddings/glove.6b/glove.6B.50d.txt")
#print(vec.size())

#print(vec[vocab["apple"],:])


print(type(vocab),type(vec))

############## 

# unk use average

# N used for number in the treebanl --> take average of 1-9 as embedding for that

# for words like bread-butter --> split on "-" and average

# otherwise if word not in embeddings use average

data = d_read.Corpus("/language-modeling-nlp1/data/penn")

train_data = data.train
valid_data = data.valid
test_data = data.test

mean_vec = torch.mean(vec,0).view(1,50)

vocab_tb = data.dictionary.word2idx.keys()



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

print("digits mean",mean_num)

print("mean words", mean_vec)





unknowns = []
for word in vocab_tb:

	if word not in vocab.keys():
		unknowns.append(word)

print("unknowns : ", unknowns)




count_uk = 0
count_num = 0
count_sp = 0


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



# exclude first row as it was just used for initialization of the tensor
embeddings = embeddings[1:][:]

print(embeddings.size())

print("numbers:",count_num,"unknowns:",count_uk,"use -  :",count_sp)










