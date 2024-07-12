#Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
#----------------

#Hyperparams
batch_size = 64
block_size = 256
steps = 10000
eval_interval =  500
learn_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_steps = 200
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2
#-----------------

#Reproducibility
torch.manual_seed(1337)
#-----------------

#Load dataset
shakespeare_filepath = r'dataset\input.txt'
with open(shakespeare_filepath, 'r', encoding='utf-8') as f:
    text = f.read()
#----------------

#Viewing dataset
len(text)
print(text[:1000])
#---------------

#Tokenization and Model vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch: i for i,ch in enumerate(chars)}
itos = { i: ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
#----------------

#Train and Validation Splits
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
#---------------

#Data Batching
x = train_data[:block_size]
y = train_data[1:block_size+1]
#--------------

#Data loading
def get_batch(split):
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            X , Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module): #Module head for a single head of self-attention
    #Single head of self-attention
    
    def __init__(self, head_size) : #Initializing our head module, that takes the head_size as a parameter
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) #Our key vector, which will be used by our tokens to communicate what it is (it's context) K - the index/tags associated with each document
        self.query = nn.Linear(n_embd, head_size, bias =False) #Our query vector, which will be the information that our token is looking for Q - the question, or search query that our token inputs
        self.value = nn.Linear(n_embd, head_size, bias =False) #Our value vector, which will be the actual information that our token wants to retrieve V - the actual contents of the documents you want to retrieve based on the question/query.
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #Register buffer/function tril so we can call it later in the masking, as it is not a parameter of the module
        
        self.dropout = nn.Dropout(dropout) #Dropout layer, where some nodes are shut off/dropped out to prevent our model of overfitting to the data
        
    def forward (self, x):
        B, T, C = x.shape #Unpacking Batch, Time , Channels from x shape
        k = self.key(x) #calling our linear Key function on x to produce (B, T, C) key vector
        q = self.query(x) #calling our linear query function on x to produce (B, T, C) query vector
        
        wei = q @ k.transpose(-2, -1) * C**-0.5 #our Weights, which will be used to aggegrate our value vector and give us information on tokens with high affinities, matches with our current token's query note - we are dividing our query and key dot product but sqrt of head_size, this is to ensure gaussian distribution in our weights---- (B, T, C) @ (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[ : T, : T ] ==0, float('-inf')) #Masking function, this is a decoder head, so tokens from the future can not communicate with past tokens (would be giving away the correct answer to our model). note - encoder heads can have future tokens communicate with past tokens, as they are used for sentiment analysis. ----- (B, T, T)
        wei = F.softmax(wei, dim=-1) #Softmax function to normalize our weights to sum to 1 ------ (B, T, T)
        
        wei = self.dropout(wei) #Dropout layer to prevent overfitting to data
        
        v = self.value(x) #calling our linear value function on x to produce (B, T, C) value vector
        out = wei @ v #Aggregation of value vector by the weights. --------- (B, T, T) @ (B, T, C) ------> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    #Multiheaded self-attention block
    
    def __init__(self, num_heads, head_size): #Initialized with number of heads and head size as parameters
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #Definition of heads in our Multiheaded Attention block
        self.proj = nn.Linear(in_features=n_embd, out_features=n_embd) #Linear projection layer, as skip connection to the Add & Norm layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #Concatenation of our heads in out as our Multihead Self-Attention block
        out = self.proj(out)
        out = self.dropout(out)
        return out
        
class FeedForward(nn.Module):
    #Feedforward Layer (A simple MLP with ReLU activation between Linear layers) / Basically convolution in our transformer
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( #a sequential container of our modules, that will add modules as they are passed below.
            nn.Linear(n_embd, 4*n_embd), #Linear layer, note - Dimensionality is 4x in the feedforward layer, and reduces back from 4x with fan_out in last layer
            nn.ReLU(), #Activation layer in our FFwd layer
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x): 
        return self.net(x)
    
class Block(nn.Module):
    #Decoder-Only transformer block
    
    def __init__(self, n_embd, n_heads): #Initialized to take number of embedding dimensions and number of heads as parameters
        super().__init__()
        head_size = n_embd // n_heads #Head size, calculated from n_embd and n_heads
        self.sa = MultiHeadAttention(n_heads, head_size) #Calling our MultiheadAttention Module as our Self-Attention in the transformer
        self.ffwd = FeedForward(n_embd) #Calling the Feedforward module, as the Feedforward layer in our transformer
        self.LN1 = nn.LayerNorm(n_embd) #Layer Normalization, note, very similar to Batch Norm, only difference is that LayerNorm is applied to rows (1th dimension)
        self.LN2 = nn.LayerNorm(n_embd) #Note we have two Layer Norms, 1 is applied before Self-attention layer and the other is applied before the FFwd layer
    
    def forward(self, x):
        x = x + self.sa(self.LN1(x)) #Our skip/residual connection, this creates a highway for the grad to flow through during backprop. Essentially, the gradient flows unchanged through x during backprop, where we fork off and do Communication on the side
        x = x + self.ffwd(self.LN2(x)) #Similar to the above comment, A skip/residual connection allowing gradient to flow through during backprop, allowing us to fork off and do Computation on the side.
        return x #Output of our transformer block
    
class BigramLM(nn.Module): #Our model module, with subclass modules nested within
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #Embedding table using nn.Embedding by PyTorch of size (Vocab_size, Vocab_size)
        self.postional_embedding_table = nn.Embedding(block_size, n_embd) #Positional encoding
        self.blocks = nn.Sequential( *[Block(n_embd=n_embd, n_heads=n_heads) for _ in range(n_layer)]) #Sequential ccontainer of our Block module, which will iterate over the number of layers and create the transformer Blocks as required
            #Block(n_embd=n_embd, n_heads=4),
            #Block(n_embd=n_embd, n_heads=4),
            #Block(n_embd=n_embd, n_heads=4),
            #Block(n_embd=n_embd, n_heads=4),
            #nn.LayerNorm(n_embd), 
        
        #Above is our blocks, Block contains our Self-Attention (communication) and Feedforward (computation) mechanisms, and it takes the number of embedding dimensions and number of heads as parameters
        
        self.LN_f = nn.LayerNorm(n_embd)
        #self.sa_heads = MultiHeadAttention(8, n_embd // 8) #Multihead Self-Attention heads for communication
        #self.ffwd = FeedForward(n_embd) #Feedforward for computation of information on per-token level
        self.lm_head = nn.Linear(n_embd, vocab_size) #Language model head
        
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        #idx and targets are both (B, T) tensors of integers
        
        tok_emb = self.token_embedding_table(idx) #(B, T, C)
        pos_emb = self.postional_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = pos_emb + tok_emb #(B, T, C)
        x = self.blocks(x)
        x = self.LN_f(x) #Final LayerNorm  (B, T, C)
        #x = self.sa_heads(x) #Feeding our token and postional embeddings into our self-attention head. ------ (B, T, C)
        #x = self.ffwd(x) #Performing the feedforward for computation ----- (B, T, C)
        logits = self.lm_head(x)#plucking our logits using lm_head function and our resulting embeddings from our self_attention head. ------- (B, T, vocab_size)
        
        if targets is None:  #Used for genration purposes, when we are not training and no ground truth targets are fed, then the loss is not calculated
            loss=None
        else:
            B, T, C = logits.shape #Unpacking the logits into Batch, Time and Channel
            logits = logits.view(B*T, C) #Stretching our logits shape, in this case multliplying B*T, C --> (B,C) tensor as required by our loss function that expects 2 dimensions (Batch, Channel)
            targets = targets.view(B*T) #Stretching our targets similar to our logits, creating one dimension of B*T --> (B)
            loss = F.cross_entropy(logits, targets) #Calculatin our loss, logits (B,C) against our targets (B)
            
        return logits , loss
        
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            idx_cond = idx[ : , -block_size : ] #cropping our index to the last block sizen index/position
            
            logits, loss = self(idx_cond)
            logits = logits[ : , -1, : ]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

model = BigramLM()
m = model.to(device)

#output_tokens = m.generate(torch.zeros((1, 1) , dtype = torch.long), max_new_tokens = 100)
#print(f'inital generation (before training): {decode(output_tokens[0].tolist())}')

optimizer = torch.optim.AdamW(m.parameters(), lr=learn_rate)

for iter in range(steps):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter} : train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())


output_tokens = torch.zeros((1,1) , dtype = torch.long, device=device)
print(decode(m.generate(output_tokens, max_new_tokens=500)[0].tolist()))