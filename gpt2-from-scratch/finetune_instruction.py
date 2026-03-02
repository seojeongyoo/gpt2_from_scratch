import json
import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. dataset
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
data = download_and_load_file(file_path, url)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]


# 2. train setting
tokenizer = tiktoken.get_encoding("gpt2")

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

num_workers = 0
batch_size = 8
torch.manual_seed(123)

customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

train_dataset = InstructionDataset(train_data, tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer)
test_dataset = InstructionDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=True, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)


# 3. model setting
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, _ = x.shape
        keys    = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = self.dropout(torch.softmax(attn_scores / self.head_dim**0.5, dim=-1))
        context = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            cfg["emb_dim"], cfg["emb_dim"],
            cfg["context_length"], cfg["drop_rate"],
            cfg["n_heads"], cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        b, t = idx.shape
        x = self.drop_emb(self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device)))
        x = self.blocks(x)
        return self.head(self.norm(x))

def load_weights_into_gpt(gpt, model_name):
    allowed_models = {
        "gpt2-small (124M)": "124M",
        "gpt2-medium (355M)": "355M",
        "gpt2-large (774M)": "774M",
        "gpt2-xl (1558M)": "1558M"
    }
    model_size = allowed_models[model_name]
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]
    os.makedirs(model_size, exist_ok=True)
    for filename in filenames:
        file_path = os.path.join(model_size, filename)
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(f"{base_url}/{model_size}/{filename}", file_path)
            print(f"다운로드 완료: {file_path}")

    import tensorflow as tf
    ckpt_path = os.path.join(model_size, "model.ckpt")
    params = {name: tf.train.load_variable(ckpt_path, name) for name, _ in tf.train.list_variables(ckpt_path)}

    gpt.pos_emb.weight = nn.Parameter(torch.tensor(params["model/wpe"]))
    gpt.tok_emb.weight = nn.Parameter(torch.tensor(params["model/wte"]))

    for b in range(len(gpt.blocks)):
        q, k, v = np.split(params[f"model/h{b}/attn/c_attn/w"][0], 3, axis=-1)
        gpt.blocks[b].att.W_query.weight = nn.Parameter(torch.tensor(q.T))
        gpt.blocks[b].att.W_key.weight   = nn.Parameter(torch.tensor(k.T))
        gpt.blocks[b].att.W_value.weight = nn.Parameter(torch.tensor(v.T))

        q_b, k_b, v_b = np.split(params[f"model/h{b}/attn/c_attn/b"], 3, axis=-1)
        gpt.blocks[b].att.W_query.bias = nn.Parameter(torch.tensor(q_b))
        gpt.blocks[b].att.W_key.bias   = nn.Parameter(torch.tensor(k_b))
        gpt.blocks[b].att.W_value.bias = nn.Parameter(torch.tensor(v_b))

        gpt.blocks[b].att.out_proj.weight = nn.Parameter(torch.tensor(params[f"model/h{b}/attn/c_proj/w"][0].T))
        gpt.blocks[b].att.out_proj.bias   = nn.Parameter(torch.tensor(params[f"model/h{b}/attn/c_proj/b"]))

        gpt.blocks[b].ff.net[0].weight = nn.Parameter(torch.tensor(params[f"model/h{b}/mlp/c_fc/w"][0].T))
        gpt.blocks[b].ff.net[0].bias   = nn.Parameter(torch.tensor(params[f"model/h{b}/mlp/c_fc/b"]))
        gpt.blocks[b].ff.net[2].weight = nn.Parameter(torch.tensor(params[f"model/h{b}/mlp/c_proj/w"][0].T))
        gpt.blocks[b].ff.net[2].bias   = nn.Parameter(torch.tensor(params[f"model/h{b}/mlp/c_proj/b"]))

        gpt.blocks[b].norm1.scale = nn.Parameter(torch.tensor(params[f"model/h{b}/ln_1/g"]))
        gpt.blocks[b].norm1.shift = nn.Parameter(torch.tensor(params[f"model/h{b}/ln_1/b"]))
        gpt.blocks[b].norm2.scale = nn.Parameter(torch.tensor(params[f"model/h{b}/ln_2/g"]))
        gpt.blocks[b].norm2.shift = nn.Parameter(torch.tensor(params[f"model/h{b}/ln_2/b"]))

    gpt.norm.scale = nn.Parameter(torch.tensor(params["model/ln_f/g"]))
    gpt.norm.shift = nn.Parameter(torch.tensor(params["model/ln_f/b"]))

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, CHOOSE_MODEL)
model.to(device)


# 4. train
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < top_logits[:, -1].unsqueeze(-1), -torch.inf, logits)
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits - logits.max(dim=-1, keepdim=True).values, dim=-1)
            idx_next = torch.multinomial(probs, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat([idx, idx_next], dim=1)
    return idx

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"에포크 {epoch+1} (Step {global_step:06d}): 훈련 손실 {train_loss:.3f}, 검증 손실 {val_loss:.3f}")

        response_text = generate(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        print(token_ids_to_text(response_text, tokenizer))

    return train_losses, val_losses, track_tokens_seen

import time

start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2
start_context = format_input(val_data[0])

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=start_context, tokenizer=tokenizer
)

end_time = time.time()
print(f"훈련 소요 시간: {(end_time - start_time) / 60:.2f}분")


# 5. generate
def generate_response(model, entry, tokenizer, device, max_new_tokens=256):
    model.eval()
    instruction_plus_input = format_input(entry)
    input_ids = text_to_token_ids(instruction_plus_input, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
    response = token_ids_to_text(token_ids, tokenizer)
    response_text = response[len(instruction_plus_input):].replace("### Response:", "").strip()
    return response_text


# 6. main
entry = test_data[0]
input_text = format_input(entry)
response = generate_response(model, entry, tokenizer, device)
print(f"입력:\n{input_text}")
print(f"\n정답:\n{entry['output']}")
print(f"\n모델 응답:\n{response}")


# 7. save
torch.save(model.state_dict(), "gpt2-medium355M-sft.pth")
print("모델이 저장되었습니다.")
