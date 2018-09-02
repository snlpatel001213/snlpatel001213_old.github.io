---
---

There are many implementtion of the sequence to sequence on internet, Then why this one even exist? Because I believe in "**What I cannot create, I do not understand"**. I have put my efforts in understanding it from scratch and finished its implementation.I have taken care and made this implementation easy to understand.This implementation of sequence to sequence is without batching. Before we go to implementation,  Let me give you some idea about

#Environment Setup

```python
!pip install torch==0.4.1 -q
!pip install torchtext -q
```

The data for this tutorial is a set of many thousands of  French  to English translation pairs. These pairs are shared from  http://www.manythings.org/anki/fra-eng.zip. 

|Fra|Eng|
|:---:|:---:|
|Go.|Va !|
|Run!|Cours !|
|Jump.|Saute.|
|Stop!|Ça suffit !|
|Stop!|Stop !|
|Stop!|Arrête-toi !|
|Wait!|Attends !|
|I am cold.|J'ai froid.|


Such 11,000+ pairs for french to English traslation are given in original dataset.


```
# !wget -c http://www.manythings.org/anki/fra-eng.zip
# !unzip fra-eng.zip
```


```
import random
import re
import string
import unicodedata

import torch
from torch.utils.data import Dataset

import time

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
```

#Data Preparation

It better to **reuse**, I am reusing official text preprocessing [code](https://github.com/pytorch/tutorials) written for seq2seq tutorial.


```
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
```

To keep track of all this we will use a helper class called Lang which has word → index (word2index) and index → word (index2word) dictionaries, as well as a count of each word word2count to use to later replace rare words.


```python
class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
```

The files are all in Unicode, to simplify we will turn Unicode characters to ASCII, make everything lowercase, and trim most punctuation.


```
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
```

To read the data file we will split the file into lines, and then split lines into pairs. The files are all English → Other Language, so if we want to translate from Other Language → English I added the `reverse` flag to reverse the pairs.


```
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('fra.txt', encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
```

Since there are a lot of example sentences and we want to train something quickly, we’ll trim the data set to only relatively short and simple sentences. Here the maximum length is 10 words (that includes ending punctuation) and we’re filtering to sentences that translate to the form “I am” or “He is” etc. (accounting for apostrophes replaced earlier).


```
eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s",
                "you are", "you re ", "we are", "we re ", "they are",
                "they re ")


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
```

The full process for preparing the data is:

- Read text file and split into lines, split lines into pairs
- Normalize text, filter by length and content
- Make word lists from sentences in pairs


```
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Input Language : ",input_lang.name, ", Number of words : " ,input_lang.n_words)
    print("Target Language : ",output_lang.name, ", Number of words : " ,output_lang.n_words)
    print("A Random Pair : ",random.choice(pairs))
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes)
    return result


def tensorFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor
```

To train, for each pair we will need an input tensor (indexes of the words in the input sentence) and target tensor (indexes of the words in the target sentence). While creating these vectors we will append the EOS token to both sequences.


```
class TextDataset(Dataset):
    def __init__(self, dataload=prepareData, lang=['eng', 'fra']):
        self.input_lang, self.output_lang, self.pairs = dataload(
            lang[0], lang[1], reverse=True)
        self.input_lang_words = self.input_lang.n_words
        self.output_lang_words = self.output_lang.n_words

    def __getitem__(self, index):
        return tensorFromPair(self.input_lang, self.output_lang,
                              self.pairs[index])

    def __len__(self):
        return len(self.pairs)
```


```
lang_dataset = TextDataset()
lang_dataloader = DataLoader(lang_dataset, shuffle=True)
```

    Reading lines...
    Input Language :  fra , Number of words :  4714
    Target Language :  eng , Number of words :  3081
    A Random Pair :  ['je n ai simplement pas faim .', 'i m just not hungry .']


One can access pairs of `French` and `English` by using iterator `lang_dataloader`.


```
for i, data in enumerate(lang_dataloader):
    in_lang, out_lang = data
    print(in_lang, out_lang)
    break
```
>tensor([[ 128,   25, 3996,  124,  210,  270,  479, 1984,    5,    1]]) tensor([[  79,   42, 2075,  161,   29,  194, 2158,    4,    1]])



```
input_size = lang_dataset.input_lang_words
hidden_size = 256
output_size = lang_dataset.output_lang_words
MAX_LENGTH = 10
```

#Some Learning

**Some basic understanding regarding PyTorch componenets used in Encoder or Decoder are  explianed below : **

##Embeddings

To learn language one must convert words to fixed size vectors. This can be done is two ways:

1. Using pretrained word vectors like Word2vec or Glove Vectors
2. Learning Vector from Scratch

Here we will be using learning Vector from Scratch, Pytorch has the **Embedding** function for the same. These embeddings will be trained as when learning takes place. Below given is the example how one can insert PyTorch Embedding layer in the model. 
For Example: We have two words in vocab "hello" and "world" and we want to have 5 dimentional vector for each word then PyTorch Embedding  can be defined in following way: [[2](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)]


```
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
```

    tensor([[ 1.7722, -1.8974,  1.3551, -0.0323, -0.5294]], grad_fn=<EmbeddingBackward>)


##Unsqueeze & Squeeze

###Unsqueeze
unsqueeze() inserts singleton dim at position given as parameter. Insert a new axis that will appear at the axis position in the expanded array shape.[[3](https://pytorch.org/docs/stable/torch.html#torch.unsqueeze)]




```
input = torch.Tensor(2, 4, 3) # input: 2 x 4 x 3
print(input.unsqueeze(0).size())
```

    torch.Size([1, 2, 4, 3])


###Squeeze
Returns a tensor with all the dimensions of input of size 1 removed. For example, if input is of shape: $(A×1×B×C×1×D)$ then the out tensor will be of shape: $(A×B×C×D)$.[[4](https://pytorch.org/docs/stable/torch.html#torch.squeeze)]



```
x = torch.zeros(2, 1, 2, 1, 2) # input: 2 x 4 x 3
y = torch.squeeze(x)
y.size()
```




    torch.Size([2, 2, 2])



##Permute

##LogSoftmax

Log Softmax applies logarithm after softmax. 

$log( exp(x_i) / exp(x).sum() )$

##TopK


```
x = torch.arange(1., 6.)
tensor([ 1.,  2.,  3.,  4.,  5.])
torch.topk(x, 3)
```

Returns the k largest elements of the given input tensor along a given dimension.

If dim is not given, the last dimension of the input is chosen.

If largest is False then the k smallest elements are returned.

A tuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor.

##GRU

#Learning Model


```
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)  # batch, hidden
        output = embedded.permute(1, 0, 2)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result
```


```
encoder_test = EncoderRNN(input_size, hidden_size)
test_encoder_input = torch.tensor([15])
encoder_hidden = encoder_test.initHidden()
encoder_output, encoder_hidden = encoder_test(test_encoder_input,encoder_hidden)
print(encoder_output.shape)
```

    torch.Size([1, 1, 256])



```
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input)  # batch, 1, hidden
        output = output.permute(1, 0, 2)  # 1, batch, hidden
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result

```


```
decoder_test = DecoderRNN(hidden_size,output_size)
decoder_hidden = decoder_test.initHidden()
test_decoder_input = torch.tensor([[13]])
decoder_output, test_hidden = decoder_test(test_decoder_input,encoder_hidden)
print(decoder_output.shape)
```

    torch.Size([1, 3081])



```
encoder = EncoderRNN(input_size, hidden_size)
decoder = DecoderRNN(hidden_size, output_size, n_layers=2)
```


```
use_attn = False
```


```
def showPlot(points):
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    x = np.arange(len(points))
    plt.plot(x, points)
    plt.show()
```

#Actual Training


```python
def train(encoder, decoder, total_epoch):
    param = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(param, lr=1e-3)
    criterion = nn.NLLLoss()
    plot_losses = []
    for epoch in range(total_epoch):
        since = time.time()
        running_loss = 0
        print_loss_total = 0
        total_loss = 0
        for i, data in enumerate(lang_dataloader):
            in_lang, out_lang = data
            
            in_lang = Variable(in_lang)  # batch=1, length
            out_lang = Variable(out_lang)

            encoder_outputs = Variable(
                torch.zeros(MAX_LENGTH, encoder.hidden_size))

            encoder_hidden = encoder.initHidden()
            for ei in range(in_lang.size(1)):
                encoder_output, encoder_hidden = encoder(
                    torch.tensor([in_lang[0][ei]]), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]

            decoder_input = Variable(torch.LongTensor([[SOS_token]]))

            decoder_hidden = encoder_hidden
            loss = 0
            for di in range(out_lang.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, torch.tensor([out_lang[0][di]]))
                    topv, topi = decoder_output.data.topk(1)
                    ni = int(topi[0][0])

                    decoder_input = Variable(torch.LongTensor([[ni]]))
                    if torch.cuda.is_available():
                        decoder_input = decoder_input.cuda()
                    if ni == EOS_token:
                        break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            print_loss_total += loss.data[0]
            total_loss += loss.data[0]
            if (i + 1) % 500 == 0:
                print('{}/{}, Loss:{:.6f}'.format(
                    i + 1, len(lang_dataloader), running_loss / 5000))
                running_loss = 0
            if (i + 1) % 100 == 0:
                plot_loss = print_loss_total / 100
                plot_losses.append(plot_loss)
                print_loss_total = 0
        during = time.time() - since
        print('Finish {}/{} , Loss:{:.6f}, Time:{:.0f}s'.format(
            epoch + 1, total_epoch, total_loss / len(lang_dataset), during))
        print()
    showPlot(plot_losses)
```


```
total_epoch = 1
train(encoder, decoder, total_epoch)
```
```
    500/11885, Loss:2.178752
    1000/11885, Loss:2.215367
    1500/11885, Loss:2.061810
    2000/11885, Loss:2.050915
    2500/11885, Loss:1.878890
    3000/11885, Loss:1.968172
    3500/11885, Loss:2.005620
    4000/11885, Loss:1.959543
    4500/11885, Loss:1.788288
    5000/11885, Loss:1.905580
    5500/11885, Loss:1.788070
    6000/11885, Loss:1.723465
    6500/11885, Loss:1.713614
    7000/11885, Loss:1.784953
    7500/11885, Loss:1.724088
    8000/11885, Loss:1.777533
    8500/11885, Loss:1.686700
    9000/11885, Loss:1.670964
    9500/11885, Loss:1.664394
    10000/11885, Loss:1.686158
    10500/11885, Loss:1.714965
    11000/11885, Loss:1.593214
    11500/11885, Loss:1.637209
    Finish 1/1 , Loss:18.277187, Time:3917s
``` 



![png](seq_2_seq_1_final_files/seq_2_seq_1_final_50_2.png)


#Evaluation


```
def evaluate(encoder, decoder, in_lang, max_length=MAX_LENGTH):
    if use_cuda:
        in_lang = in_lang.cuda()
    input_variable = Variable(in_lang)
    input_variable = input_variable.unsqueeze(0)
    input_length = input_variable.size(1)
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(torch.tensor([input_variable[0][ei]]),encoder_hidden)
        
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni =  int(topi[0][0])
            if ni == EOS_token:
                break
            else:
                decoded_words.append(lang_dataset.output_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
            pair_idx = random.choice(list(range(len(lang_dataset))))
            pair = lang_dataset.pairs[pair_idx]
            in_lang, out_lang = lang_dataset[pair_idx]
            output_words = evaluate(encoder, decoder, in_lang)
            output_sentence = ' '.join(output_words)
            print('Input : ', pair[0],' | Desired Output : ', pair[1],' | Generated Output : ', output_sentence)
```


```
evaluateRandomly(encoder, decoder)
```

    Input :  elle joue a la poupee .  | Desired Output :  she is playing with a doll .  | Generated Output :  she is playing the . .
    
    Input :  vous etes trop maigre .  | Desired Output :  you re too skinny .  | Generated Output :  you re too loud .
    
    Input :  tu es precis .  | Desired Output :  you re precise .  | Generated Output :  you re productive .
    
    Input :  je suis impatiente de te voir bientot .  | Desired Output :  i m looking forward to seeing you soon .  | Generated Output :  i am looking forward seeing seeing seeing . .
    
    Input :  je suis interesse par la ceramique orientale .  | Desired Output :  i m interested in oriental pottery .  | Generated Output :  i am interested in in in .
    
    Input :  elle est en danger .  | Desired Output :  she s in danger .  | Generated Output :  she is in danger .
    
    Input :  je crains de t avoir offensee .  | Desired Output :  i m afraid i ve offended you .  | Generated Output :  i m afraid i ve ve you . . .
    
    Input :  nous sommes desoles pour le derangement .  | Desired Output :  we are sorry for the inconvenience .  | Generated Output :  we re the for the . . .
    
    Input :  je ne parle pas avec toi .  | Desired Output :  i m not talking to you .  | Generated Output :  i m not not you you you .
    
    Input :  je suis toujours amoureuse de mary .  | Desired Output :  i m still in love with mary .  | Generated Output :  i m always proud of . .
    


    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.


# https://github.com/L1aoXingyu/seq2seq-translation
