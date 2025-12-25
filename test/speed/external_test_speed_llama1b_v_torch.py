import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import unittest
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
np.set_printoptions(linewidth=160)
from extra.models.llama import Transformer as TinygradTransformer
from tinygrad import Tensor, Device
from tinygrad.nn.state import get_state_dict
from tinygrad.helpers import colorize_float, getenv
from tabulate import tabulate
torch.set_num_threads(1)

TORCHCOMPILE = bool(int(getenv("TORCHCOMPILE", 1)))
CNT = getenv("CNT", 10)
TEMPERATURE = 0  # greedy decoding

# llama 1B config taken from examples/llama3.py
LLAMA_CONFIG = {
  'dim': 2048,
  'n_heads': 32,
  'n_kv_heads': 8,
  'n_layers': 16,
  'hidden_dim': 8192,
  'vocab_size': 128256,
  'norm_eps': 1e-5,
  'max_context': 128
}

# +----- Start of Torch Port of extras/models/llama.py -----+

def torch_precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
  freqs = torch.arange(end).unsqueeze(1) * freqs.unsqueeze(0)
  return torch.stack([freqs.cos(), freqs.sin()], dim=-1).reshape(1, end, 1, dim//2, 2)

def torch_complex_mult(A, c, d):
  a, b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return torch.cat([ro, co], dim=-1)

def torch_apply_rotary_emb(xq, xk, freqs_cis):
  xq = xq.reshape(*xq.shape[:-1], -1, 2)
  xk = xk.reshape(*xk.shape[:-1], -1, 2)
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  xq_out = torch_complex_mult(xq, c, d)
  xk_out = torch_complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)

def torch_repeat_kv(x, n_rep):
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1: return x
  return x.repeat(1, 1, 1, n_rep).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class TorchRMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps
  def forward(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class TorchAttention(nn.Module):
  def __init__(self, dim, n_heads, n_kv_heads, max_context):
    super().__init__()
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = dim // n_heads
    self.n_rep = n_heads // n_kv_heads
    self.max_context = max_context

    self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    self.register_buffer('cache_kv', torch.zeros(2, 1, max_context, n_kv_heads, self.head_dim))

  def forward(self, x, start_pos, freqs_cis):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
    xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

    xq, xk = torch_apply_rotary_emb(xq, xk, freqs_cis)

    self.cache_kv[:, :, start_pos:start_pos+seqlen, :, :] = torch.stack([xk, xv])
    keys = self.cache_kv[0, :, :start_pos+seqlen, :, :]
    values = self.cache_kv[1, :, :start_pos+seqlen, :, :]

    keys, values = torch_repeat_kv(keys, self.n_rep), torch_repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

    mask = None
    if seqlen > 1:
      mask = torch.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), device=x.device).triu(start_pos+1)

    attn = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask)
    attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.wo(attn)

class TorchFeedForward(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))

def torch_sample(logits, temp):
  """Torch equivalent of tinygrad's sample function for greedy decoding."""
  assert logits.dim() == 1, "only works on 1d tensors"
  if temp < 1e-6:
    return logits.argmax()
  return torch.multinomial(F.softmax(logits / temp, dim=-1), 1).squeeze()

class TorchTransformerBlock(nn.Module):
  def __init__(self, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context):
    super().__init__()
    self.attention = TorchAttention(dim, n_heads, n_kv_heads, max_context)
    self.feed_forward = TorchFeedForward(dim, hidden_dim)
    self.attention_norm = TorchRMSNorm(dim, norm_eps)
    self.ffn_norm = TorchRMSNorm(dim, norm_eps)

  def forward(self, x, start_pos, freqs_cis):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
    return h + self.feed_forward(self.ffn_norm(h))

class TorchTransformer(nn.Module):
  def __init__(self, dim, hidden_dim, n_heads, n_layers, norm_eps, vocab_size, n_kv_heads, max_context):
    super().__init__()
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.layers = nn.ModuleList([
      TorchTransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context)
      for _ in range(n_layers)
    ])
    self.norm = TorchRMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    self.register_buffer('freqs_cis', torch_precompute_freqs_cis(dim // n_heads, max_context * 2))

  def forward(self, tokens, start_pos, temperature=0.0):
    seqlen = tokens.shape[1]
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis[:, start_pos:start_pos+seqlen, :, :, :]
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis)
    logits = self.output(self.norm(h))
    return torch_sample(logits[:, -1, :].flatten(), temperature)

# +----- End of Torch Port of extras/models/llama.py -----+

def benchmark_torch(model, start_tok, warmup=10, iters=CNT):
  cache_defeat = np.zeros((2048, 2048))
  cache_defeat += 1

  toks = []
  with torch.no_grad():
    last_tok = start_tok
    for i in range(warmup):
      last_tok = model(torch.tensor([[last_tok]]), i, TEMPERATURE).item()

  times = []
  with torch.no_grad():
    for i in range(iters):
      st = time.perf_counter()
      last_tok = model(torch.tensor([[last_tok]]), warmup + i, TEMPERATURE).item()
      times.append(time.perf_counter() - st)
      toks.append(last_tok)
  return times, toks

def benchmark_tinygrad(model, start_tok, warmup=10, iters=CNT):
  cache_defeat = np.zeros((2048, 2048))
  cache_defeat += 1
  
  toks = []
  tok_tensor = Tensor([[start_tok]]).realize()
  for i in range(warmup):
    last_tok = model(tok_tensor, i, TEMPERATURE).item()
    tok_tensor.assign(Tensor([[last_tok]])).realize()

  times = []
  for i in range(iters):
    st = time.perf_counter()
    last_tok = model(tok_tensor, warmup + i, TEMPERATURE).item()
    times.append(time.perf_counter() - st)
    tok_tensor.assign(Tensor([[last_tok]])).realize()
    toks.append(last_tok)
  return times, toks

def copy_weights_torch_to_tinygrad(torch_model, tiny_model):
  tiny_state = get_state_dict(tiny_model)
  torch_state = torch_model.state_dict()
  for name, param in torch_state.items():
    if 'freqs_cis' in name or 'cache_kv' in name:
      continue
    if name in tiny_state:
      tiny_state[name].assign(Tensor(param.numpy()))

class BaseLlamaTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print(f"\nTinygrad device: {Device.DEFAULT}")
    print(f"Config: {LLAMA_CONFIG}")
    print(f"Temperature: {TEMPERATURE} (greedy decoding)")

    cls.torch_model = TorchTransformer(**LLAMA_CONFIG)
    cls.tiny_model = TinygradTransformer(**LLAMA_CONFIG, jit=True)

    torch_model_uncompiled = TorchTransformer(**LLAMA_CONFIG)
    torch_model_uncompiled.load_state_dict(cls.torch_model.state_dict())
    copy_weights_torch_to_tinygrad(torch_model_uncompiled, cls.tiny_model)

    cls.torch_ref = torch_model_uncompiled
    cls.start_tok = 1  # starting token for generation

  def reset_kv(self):
    for layer in self.torch_ref.layers:
      layer.attention.cache_kv.zero_()
    for layer in self.torch_model.layers:
      layer.attention.cache_kv.zero_()
    for layer in self.tiny_model.layers:
      if hasattr(layer.attention, 'cache_kv'):
        delattr(layer.attention, 'cache_kv')

class TestLlamaBenchmark(BaseLlamaTest):
  def test_benchmark(self):
    self.reset_kv()

    if TORCHCOMPILE:
      print("\nCompiling torch model...")
      torch_model = torch.compile(self.torch_model)
    else:
      torch_model = self.torch_ref

    torch_times, torch_toks = benchmark_torch(torch_model, self.start_tok)
    self.reset_kv()
    tiny_times, tiny_toks = benchmark_tinygrad(self.tiny_model, self.start_tok)

    # correctness: assert generated sequences match
    self.assertEqual(torch_toks, tiny_toks, f"Generated sequences differ: torch={torch_toks[:10]}... tiny={tiny_toks[:10]}...")

    table_data = [[i+1, f"{torch_times[i]*1000:.2f}", f"{tiny_times[i]*1000:.2f}", colorize_float(tiny_times[i]/torch_times[i])] for i in range(len(torch_times))]
    print("\nBenchmark results (greedy decoding):")
    print(tabulate(table_data, headers=["Token", "Torch (ms)", "Tinygrad (ms)", "Ratio"], tablefmt="simple"))

    print(f"\nAverage times: Torch={np.mean(torch_times)*1000:.2f}ms, Tinygrad={np.mean(tiny_times)*1000:.2f}, Ratio={np.mean(tiny_times)/np.mean(torch_times):.2f}")

if __name__ == "__main__":
  unittest.main(verbosity=2)
