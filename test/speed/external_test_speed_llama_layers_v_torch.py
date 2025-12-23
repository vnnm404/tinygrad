import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import unittest
import torch
torch.set_num_threads(1)
import time
import numpy as np
np.set_printoptions(linewidth=160)
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
from tinygrad.helpers import colorize_float, getenv, CI

# LLaMA 1B config
DIM = 2048
N_HEADS = 32
N_KV_HEADS = 8
HIDDEN_DIM = 8192
VOCAB_SIZE = 128256
HEAD_DIM = DIM // N_HEADS

torch_dt = torch.float16 if getenv("HALF", 0) else torch.float32
torch_device = torch.device('mps' if getenv("MPS", 0) else ('cuda' if getenv("TORCHCUDA", 0) else 'cpu'))
if str(torch_device) == "mps":
  import torch.mps
  def sync(): torch.mps.synchronize()
elif str(torch_device) == "cuda":
  import torch.cuda
  def sync(): torch.cuda.synchronize()
else:
  def sync(): pass

save_ops, save_mem = 0, 0
CNT = getenv("CNT", 8)
def helper_test_speed(f1, *args):
  global save_ops, save_mem
  ets = []
  ret = None
  cache_defeat = np.zeros((2048,2048))
  for i in range(CNT):
    del ret

    # operation cache defeats
    args = [(x+1).realize() if isinstance(x, Tensor) else (None if x is None else (x+1)) for x in args]
    args = [(x-1).realize() if isinstance(x, Tensor) else (None if x is None else (x-1)) for x in args]

    # force syncing
    [x.numpy() if isinstance(x, Tensor) or str(torch_device) == "cpu" else x.cpu().numpy() for x in args if x is not None]

    # clear 32MB global memory cache (CPU and global memory only)
    cache_defeat += 1

    # manual pre sync
    if isinstance(args[0], Tensor):
      local_device = Device[args[0].device]
      local_device.synchronize()
    else: sync()

    GlobalCounters.global_ops = 0
    GlobalCounters.global_mem = 0
    st = time.perf_counter()
    ret = f1(*args)
    if isinstance(ret, Tensor): local_device.synchronize()
    else: sync()
    et = (time.perf_counter() - st) * 1000
    if i >= 1: ets.append(et)
    if GlobalCounters.global_ops:
      save_ops, save_mem = GlobalCounters.global_ops, GlobalCounters.global_mem
  return ret.numpy() if isinstance(ret, Tensor) else ret.cpu().numpy(), np.min(ets)

def helper_test_generic_square(name, N, f1, f2, onearg=False):
  torch.manual_seed(0)
  torch_a = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device)
  torch_b = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device) if not onearg else None

  tiny_a = Tensor(torch_a.cpu().numpy())
  tiny_b = Tensor(torch_b.cpu().numpy()) if not onearg else None

  helper_test_generic(f"{name:30s} {N:5d}x{N:5d}", f1, (torch_a, torch_b), TinyJit(f2), (tiny_a, tiny_b))

prefix = None
def helper_test_generic(name, f1, f1_args, f2, f2_args):
  global prefix
  with torch.no_grad():
    val_torch, et_torch = helper_test_speed(f1, *f1_args)
  val_tinygrad, et_tinygrad = helper_test_speed(f2, *f2_args)

  desc = "faster" if et_torch > et_tinygrad else "slower"
  flops = save_ops*1e-6
  mem = save_mem*1e-6
  print(("\r" if not CI else "")+f"{name:42s} {et_torch:7.2f} ms ({flops/et_torch:9.2f} GFLOPS {mem/et_torch:7.2f} GB/s) in torch, {et_tinygrad:7.2f} ms ({flops/et_tinygrad:9.2f} GFLOPS {mem/et_tinygrad:7.2f} GB/s) in tinygrad, {colorize_float(et_tinygrad/et_torch)} {desc} {flops:10.2f} MOPS {mem:8.2f} MB")  # noqa: E501
  atol, rtol = (1e-2, 1e-2) if torch_dt == torch.float16 else (1e-3, 1e-3)
  np.testing.assert_allclose(val_tinygrad, val_torch, atol=atol, rtol=rtol)

@unittest.skipIf(getenv("BIG") == 0, "no big tests")
@unittest.skipIf(getenv("MOCKGPU"), "no MOCKGPUs")
class TestBigSpeed(unittest.TestCase):
  def test_add(self):
    def f(a, b): return a+b
    helper_test_generic_square('add', 8192, f, f)
  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 8192, f, f, onearg=True)
  def test_gemm_2048(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 2048, f, f)
  def test_gemm_4096(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 4096, f, f)

@unittest.skipIf(getenv("BIG") == 1, "only big tests")
@unittest.skipIf(getenv("MOCKGPU"), "no MOCKGPUs")
class TestSpeed(unittest.TestCase):
  def test_mul_sum(self):
    def f(a, b): return (a*b).sum()
    helper_test_generic_square('mul_sum', 4096, f, f)

  def test_add_big(self):
    for N in [1024, 4096]:
      def f(a, b): return a + b
      helper_test_generic_square('add', N, f, f)

  def test_gemm(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 1024, f, f)

  def test_gemm_small(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 256, f, f)

  def test_silu(self):
    def f1(a, b): return torch.nn.functional.silu(a)
    def f2(a, b): return a.silu()
    helper_test_generic_square('silu', 2048, f1, f2, onearg=True)

  def test_softmax(self):
    def f1(a, b): return torch.nn.functional.softmax(a, dim=-1)
    def f2(a, b): return a.softmax(axis=-1)
    helper_test_generic_square('softmax', 2048, f1, f2, onearg=True)

  def test_rmsnorm(self):
    eps = 1e-5
    def f1(a, b): return a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + eps)
    def f2(a, b): return a * (a.square().mean(-1, keepdim=True) + eps).rsqrt()
    helper_test_generic_square('rmsnorm', 2048, f1, f2, onearg=True)

  def test_sdpa(self):
    N_HEADS, SEQ_LEN, HEAD_DIM = 32, 64, 64
    torch.manual_seed(0)
    torch_q = (torch.rand(1, N_HEADS, 1, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_k = (torch.rand(1, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_v = (torch.rand(1, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_q, tiny_k, tiny_v = Tensor(torch_q.cpu().numpy()), Tensor(torch_k.cpu().numpy()), Tensor(torch_v.cpu().numpy())
    def f1(q, k, v): return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    def f2(q, k, v): return q.scaled_dot_product_attention(k, v)
    helper_test_generic(f"sdpa heads:{N_HEADS} seq:{SEQ_LEN} dim:{HEAD_DIM}", f1, (torch_q, torch_k, torch_v),
                        TinyJit(f2), (tiny_q, tiny_k, tiny_v))

  def test_argmax(self):
    N = 128256  # vocab size
    torch.manual_seed(0)
    torch_a = (torch.rand(N, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_a = Tensor(torch_a.cpu().numpy())
    def f1(a): return a.argmax()
    def f2(a): return a.argmax()
    helper_test_generic(f"argmax {N}", f1, (torch_a,), TinyJit(f2), (tiny_a,))

  def test_embedding(self):
    torch.manual_seed(0)
    torch_weight = (torch.rand(VOCAB_SIZE, DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_idx = torch.tensor([[42]]).to(torch_device)
    tiny_weight = Tensor(torch_weight.cpu().numpy())
    tiny_idx = Tensor([[42]])
    def f1(w, idx): return w[idx]
    def f2(w, idx): return w[idx]
    helper_test_generic(f"embedding {VOCAB_SIZE}x{DIM}", f1, (torch_weight, torch_idx), TinyJit(f2), (tiny_weight, tiny_idx))

  def test_linear_ffn_up(self):
    torch.manual_seed(0)
    torch_x = (torch.rand(1, 1, DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_w = (torch.rand(DIM, HIDDEN_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_x = Tensor(torch_x.cpu().numpy())
    tiny_w = Tensor(torch_w.cpu().numpy())
    def f(x, w): return x @ w
    helper_test_generic(f"linear_ffn_up {DIM}->{HIDDEN_DIM}", f, (torch_x, torch_w), TinyJit(f), (tiny_x, tiny_w))

  def test_linear_ffn_down(self):
    torch.manual_seed(0)
    torch_x = (torch.rand(1, 1, HIDDEN_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_w = (torch.rand(HIDDEN_DIM, DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_x = Tensor(torch_x.cpu().numpy())
    tiny_w = Tensor(torch_w.cpu().numpy())
    def f(x, w): return x @ w
    helper_test_generic(f"linear_ffn_down {HIDDEN_DIM}->{DIM}", f, (torch_x, torch_w), TinyJit(f), (tiny_x, tiny_w))

  def test_linear_output(self):
    torch.manual_seed(0)
    torch_x = (torch.rand(1, 1, DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_w = (torch.rand(DIM, VOCAB_SIZE, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_x = Tensor(torch_x.cpu().numpy())
    tiny_w = Tensor(torch_w.cpu().numpy())
    def f(x, w): return x @ w
    helper_test_generic(f"linear_output {DIM}->{VOCAB_SIZE}", f, (torch_x, torch_w), TinyJit(f), (tiny_x, tiny_w))

  def test_rope(self):
    SEQ_LEN = 64
    torch.manual_seed(0)
    torch_x = (torch.rand(1, SEQ_LEN, N_HEADS, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_cos = torch.rand(1, SEQ_LEN, 1, HEAD_DIM, dtype=torch_dt).to(torch_device)
    torch_sin = torch.rand(1, SEQ_LEN, 1, HEAD_DIM, dtype=torch_dt).to(torch_device)
    tiny_x = Tensor(torch_x.cpu().numpy())
    tiny_cos = Tensor(torch_cos.cpu().numpy())
    tiny_sin = Tensor(torch_sin.cpu().numpy())
    def rotate_half_torch(x):
      x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
      return torch.cat((-x2, x1), dim=-1)
    def rotate_half_tiny(x):
      x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
      return (-x2).cat(x1, dim=-1)
    def f1(x, cos, sin): return x * cos + rotate_half_torch(x) * sin
    def f2(x, cos, sin): return x * cos + rotate_half_tiny(x) * sin
    helper_test_generic(f"rope seq={SEQ_LEN} heads={N_HEADS}", f1, (torch_x, torch_cos, torch_sin),
                        TinyJit(f2), (tiny_x, tiny_cos, tiny_sin))

  def test_elementwise_mul(self):
    torch.manual_seed(0)
    torch_a = (torch.rand(1, 1, HIDDEN_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_b = (torch.rand(1, 1, HIDDEN_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_a = Tensor(torch_a.cpu().numpy())
    tiny_b = Tensor(torch_b.cpu().numpy())
    def f(a, b): return a * b
    helper_test_generic(f"elementwise_mul {HIDDEN_DIM}", f, (torch_a, torch_b), TinyJit(f), (tiny_a, tiny_b))

  def test_kv_cache_update(self):
    SEQ_LEN = 64
    torch.manual_seed(0)
    torch_cache = (torch.rand(1, N_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    torch_new = (torch.rand(1, N_KV_HEADS, 1, HEAD_DIM, dtype=torch_dt) - 0.5).to(torch_device)
    tiny_cache = Tensor(torch_cache.cpu().numpy())
    tiny_new = Tensor(torch_new.cpu().numpy())
    def f1(cache, new): return torch.cat([cache, new], dim=2)
    def f2(cache, new): return cache.cat(new, dim=2)
    helper_test_generic(f"kv_cache_update seq={SEQ_LEN}+1", f1, (torch_cache, torch_new), TinyJit(f2), (tiny_cache, tiny_new))

if __name__ == '__main__':
  unittest.main()
