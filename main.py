import torch
import torch.nn.functional as F
import torch.nn as nn
from tokenizer import Tokenizer

mps_device = torch.device("mps")

embed_size = 192
context_size = 128
batch_size = 64
eval_iterations = 10
epochs = 5000
dropout = 0.2


class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()

        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        query = self.query(x)
        key = self.key(x)

        wei = query @ key.transpose(-2, -1) * embed_size**-0.5
        wei = wei.masked_fill(self.tril == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        value = self.value(x)
        return wei @ value  # context_size, head_size


class MutliHeadAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(embed_size // num_heads) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class Block(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.attention_head = MutliHeadAttention(embed_size, num_heads)
        self.feedForward = FeedForward(embed_size)

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor):
        x = self.layer_norm1(x)
        x = x + self.attention_head(x)
        x = self.layer_norm2(x)
        x = x + self.feedForward(x)
        return x


class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(context_size, embed_size).to(
            mps_device
        )

        self.blocks = nn.Sequential(
            Block(embed_size, 6),
            Block(embed_size, 6),
            Block(embed_size, 6),
            Block(embed_size, 6),
            Block(embed_size, 6),
            Block(embed_size, 6),
            nn.LayerNorm(embed_size),
        )

        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        token_embeddings = self.embedding(x)
        pos_embeddings = self.positional_embedding(
            torch.arange(context_size).to(mps_device)
        )

        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.linear(x)  # context_size, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(x, targets)

        return x, loss

    def generate(self, length: int):
        generated = torch.zeros(1, context_size).to(mps_device).to(torch.long)

        for _ in range(length):
            context = generated[:, -context_size:]

            logits, _ = self(context)
            prediction_logits = logits[:, -1, :]
            probabilities = F.softmax(prediction_logits, dim=-1)
            prediction = torch.multinomial(probabilities, num_samples=1)

            generated = torch.cat((generated, prediction), dim=1)
            print(tokenizer.decode([generated.tolist()[0][-1]]), end="", flush=True)

        return generated.tolist()[0][context_size:]


def get_batch(data, batch_size):
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i + context_size] for i in ix]).to(mps_device)
    y = torch.stack([data[i + 1 : i + context_size + 1] for i in ix]).to(mps_device)
    return x.to(torch.long), y.to(torch.long)


def get_data_splits(data, split_ratio):
    split = int(len(data) * split_ratio)
    train_data = torch.Tensor(tokenizer.encode(data[:split])).to(mps_device)
    test_data = torch.Tensor(tokenizer.encode(data[split:])).to(mps_device)
    return train_data, test_data


@torch.no_grad()
def evaluate_loss():
    model.eval()

    for split in ["train", "test"]:
        total_loss = 0

        for _ in range(eval_iterations):
            x, y = get_batch(train_data if split == "train" else test_data, batch_size)
            _, loss = model(x, y)
            total_loss += loss.item()

        if split == "train":
            train_loss = total_loss / (eval_iterations)
        else:
            test_loss = total_loss / (eval_iterations)

    model.train()
    return train_loss, test_loss


def train(model, epochs):
    model.train()
    for i in range(epochs):
        x, targets = get_batch(train_data, batch_size)

        _, loss = model(x, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(loss.item())


with open("data.txt") as file:
    text_data = file.read()


tokenizer = Tokenizer()
# print(tokenizer.decode(tokenizer.encode("\nhello\n: HOW ARE you!!")))
# vocabulary = tokenizer.train(text_data, 200)
# tokenizer.save("token.vocab")
# print(tokenizer.decode(tokenizer.encode("hello: HOW ARE you!!")))

tokenizer.load("token.vocab")

model = Model(len(tokenizer.vocab)).to(mps_device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("nb parameters:", sum([p.nelement() for p in model.parameters()]))

train_data, test_data = get_data_splits(text_data, 0.9)
train(model, epochs)
torch.save(model.state_dict(), "gpt-regex-2.pt")

# model.load_state_dict(torch.load("gpt-regex-2.pt", map_location=torch.device(mps_device)))
# generated = model.generate(500)
# print(tokenizer.decode(generated))
