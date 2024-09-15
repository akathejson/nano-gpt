## Bigram Language Model

import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)
# mps_device = torch.device("mps")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    n_embed = 384
    n_head = 6
    num_layers = 6

    dropout = 0.2
else:
    batch_size = 32
    block_size = 8
    max_iters = 5000
    eval_interval = 500
    learning_rate = 1e-3
    eval_iters = 200
    n_embed = 32
    num_layers = 4
    dropout = 0.2
# data
with open('input.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab = ''.join(chars)
vocab_size = len(vocab)    

def get_batch(split):
    data = train_data if split =="train" else test_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiHeadAttention(nn.Module):   
    """ Multi-Head Attention in parallel """

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # from trasnformer paper 
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block : commuication followed by computation"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head   
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.token_embedding_tabel = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, 4) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)

        # self.sa_heads = MultiHeadAttention(4, n_embed//4)    
        # self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_tabel(idx) # B x T x C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T x C
        x = pos_emb + tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # B x T x VocabSize
        # print("IDX:",idx)
        # print("Logits: ",logits.shape)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is B x T  array of indexes in the current context
        for _ in range(max_new_tokens):
            #get the predictions for the next token
            idx_cond = idx[:, -block_size:] # B x T
            logits, loss = self(idx_cond)
            # focus only on the last step
            logits = logits[:,-1, :] # B x C
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=-1)    
        return idx


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed,head_size, bias=False)
        self.value = nn.Linear(n_embed,head_size, bias=False)
        self. register_buffer('tril', torch. tril(torch. ones (block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T , C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # B x T x T
        wei = self.dropout(wei)   
        v = self.value(x)
        out = wei @ v

        return out

if __name__ == "__main__":

    

    stoi = {ch : i for i, ch in enumerate(chars)}
    itos = {i : ch for i, ch in enumerate(chars)}   

    encode = lambda x: [stoi[c] for c in x]
    decode  = lambda x: ''.join([itos[i] for i in x])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(.9* len(data))
    train_data = data[:n]
    test_data = data[n:]

    # xb, yb = get_batch('train')

    # logits,loss = m(xb,yb)
    # print(logits.shape)
    # print(loss)
    # print(decode(m.generate(idx= torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    model = BigramLanguageModel() 
    m = model.to(device)
    optimiser = torch.optim.AdamW(m.parameters(), lr=learning_rate)  

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    msg = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    # save msg to file
    with open('output.txt', 'w') as f:
        f.write(msg)
    print(msg)
    # print(decode(m.generate(idx= torch.zeros((1,1), dtype=torch.long), max_new_tokens=300)[0].tolist()))