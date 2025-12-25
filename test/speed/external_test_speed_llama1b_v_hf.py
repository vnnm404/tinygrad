import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import unittest
import time
import numpy as np
import torch
torch.set_num_threads(1)
np.set_printoptions(linewidth=160)
from transformers import LlamaForCausalLM, LlamaConfig, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper
from extra.models.llama import Transformer as TinygradTransformer, convert_from_huggingface
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.nn.state import get_state_dict, load_state_dict
from tinygrad.helpers import colorize_float, getenv
from tabulate import tabulate
from tqdm import tqdm

# plot flag
PLOT = bool(int(getenv("PLOT", 0)))

TORCHCOMPILE = bool(int(getenv("TORCHCOMPILE", 1)))
CNT = getenv("CNT", 10)
WARMUP = 10

# llama 1B config
LLAMA_CONFIG = {
  'dim': 2048,
  'n_heads': 32,
  'n_kv_heads': 8,
  'n_layers': 16,
  'hidden_dim': 8192,
  'vocab_size': 128256,
  'norm_eps': 1e-5,
  'rope_theta': 500000,
  'max_context': WARMUP + CNT
}

# sampling parameters
TEMPERATURE = getenv("TEMPERATURE", 0.0)
TOP_K = 5
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

def create_hf_config():
  """Create HuggingFace LlamaConfig matching our LLAMA_CONFIG."""
  return LlamaConfig(
    vocab_size=LLAMA_CONFIG['vocab_size'],
    hidden_size=LLAMA_CONFIG['dim'],
    intermediate_size=LLAMA_CONFIG['hidden_dim'],
    num_hidden_layers=LLAMA_CONFIG['n_layers'],
    num_attention_heads=LLAMA_CONFIG['n_heads'],
    num_key_value_heads=LLAMA_CONFIG['n_kv_heads'],
    rms_norm_eps=LLAMA_CONFIG['norm_eps'],
    rope_theta=LLAMA_CONFIG['rope_theta'],
    max_position_embeddings=LLAMA_CONFIG['max_context'] * 2,
    use_cache=True,
  )

def benchmark_hf(model, start_tok, warmup=WARMUP, iters=CNT):
  cache_defeat = np.zeros((2048, 2048))
  cache_defeat += 1

  device = next(model.parameters()).device
  toks = []

  # build logits processor list using HF's native warpers
  logits_processor = LogitsProcessorList()
  if TEMPERATURE > 0: logits_processor.append(TemperatureLogitsWarper(TEMPERATURE))
  if TOP_K > 0: logits_processor.append(TopKLogitsWarper(TOP_K))
  if TOP_P > 0: logits_processor.append(TopPLogitsWarper(TOP_P))

  def sample_hf(logits, input_ids):
    if TEMPERATURE < 1e-6: return logits.argmax(dim=-1).item()
    scores = logits_processor(input_ids, logits)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, 1).item()

  with torch.no_grad():
    past_key_values = None
    input_ids = torch.tensor([[start_tok]], device=device)
    for i in range(warmup):
      outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
      past_key_values = outputs.past_key_values
      logits = outputs.logits[:, -1, :]
      next_tok = sample_hf(logits, input_ids)
      input_ids = torch.tensor([[next_tok]], device=device)

  times = []
  with torch.no_grad():
    for i in tqdm(range(iters), desc="HF Decoding"):
      st = time.perf_counter()
      outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
      past_key_values = outputs.past_key_values
      logits = outputs.logits[:, -1, :]
      next_tok = sample_hf(logits, input_ids)
      times.append(time.perf_counter() - st)
      input_ids = torch.tensor([[next_tok]], device=device)
      toks.append(next_tok)
  return times, toks

def benchmark_tinygrad(model, start_tok, warmup=WARMUP, iters=CNT):
  cache_defeat = np.zeros((2048, 2048))
  cache_defeat += 1

  toks = []
  tok_tensor = Tensor([[start_tok]]).realize()
  for i in range(warmup):
    last_tok = model(tok_tensor, i, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
    tok_tensor.assign(Tensor([[last_tok]])).realize()

  times = []
  mems = []
  for i in tqdm(range(iters), desc="Tinygrad Decoding"):
    GlobalCounters.reset()
    st = time.perf_counter()
    last_tok = model(tok_tensor, warmup + i, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).item()
    elapsed = time.perf_counter() - st
    times.append(elapsed)
    # GlobalCounters.global_mem is bytes accessed; record for GB/s computation
    mems.append(getattr(GlobalCounters, "global_mem", 0))
    tok_tensor.assign(Tensor([[last_tok]])).realize()
    toks.append(last_tok)
  return times, toks, mems

def copy_weights_hf_to_tinygrad(hf_model, tiny_model):
  """Copy weights from HuggingFace model to tinygrad model using convert_from_huggingface."""
  hf_state = {k: Tensor(v.cpu().numpy()) for k, v in hf_model.state_dict().items()}
  tiny_weights = convert_from_huggingface(hf_state, LLAMA_CONFIG['n_layers'], LLAMA_CONFIG['n_heads'], LLAMA_CONFIG['n_kv_heads'])
  load_state_dict(tiny_model, tiny_weights, strict=False)

class BaseLlamaTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print(f"\nTinygrad device: {Device.DEFAULT}")
    print(f"Config: {LLAMA_CONFIG}")
    print(f"Temperature: {TEMPERATURE} ({'sampling' if TEMPERATURE > 0 else 'greedy decoding'})")
    print("Using HuggingFace transformers LlamaForCausalLM")

    # create HuggingFace model with random weights
    hf_config = create_hf_config()
    print("Creating HuggingFace model with random weights...")
    cls.hf_model = LlamaForCausalLM(hf_config)
    cls.hf_model.eval()

    # create tinygrad model
    print("Creating tinygrad model...")
    cls.tiny_model = TinygradTransformer(**LLAMA_CONFIG, jit=True)

    # copy weights from HF to tinygrad
    print("Copying weights from HuggingFace to tinygrad...")
    copy_weights_hf_to_tinygrad(cls.hf_model, cls.tiny_model)

    cls.start_tok = 1  # starting token for generation

  def reset_kv(self):
    # reset tinygrad kv cache
    for layer in self.tiny_model.layers:
      if hasattr(layer.attention, 'cache_kv'):
        delattr(layer.attention, 'cache_kv')

class TestLlamaBenchmark(BaseLlamaTest):
  @unittest.skipIf(TEMPERATURE > 0, "Skipping correctness test when sampling (TEMPERATURE > 0)")
  def test_correctness(self):
    """Test that HuggingFace and tinygrad produce similar logits."""
    self.reset_kv()

    # run a short sequence through both models and compare logits
    test_tokens = [1, 100, 200, 300]
    input_ids = torch.tensor([test_tokens])

    with torch.no_grad():
      hf_outputs = self.hf_model(input_ids, use_cache=False)
      hf_logits = hf_outputs.logits[0, -1, :].numpy()

    tiny_logits = self.tiny_model.forward(
      Tensor([test_tokens]).reshape(1, -1), 0, temperature=float('nan'), top_k=0, top_p=0, alpha_f=0, alpha_p=0
    )[0, -1, :].numpy()

    # check correlation - should be very high if weights are copied correctly
    correlation = np.corrcoef(hf_logits.flatten(), tiny_logits.flatten())[0, 1]
    print(f"\nLogits correlation: {correlation:.6f}")

    # with random weights, numerical differences can accumulate, so we check correlation
    self.assertGreater(correlation, 0.99, f"Logits correlation too low: {correlation}")

  def test_benchmark(self):
    self.reset_kv()

    if TORCHCOMPILE:
      print("\nCompiling torch model...")
      hf_model = torch.compile(self.hf_model)
    else:
      hf_model = self.hf_model

    print("Running benchmarks...")

    # HuggingFace benchmark (creates fresh KV cache each run)
    hf_times, hf_toks = benchmark_hf(hf_model, self.start_tok)

    # tinygrad benchmark
    self.reset_kv()
    tiny_times, tiny_toks, tiny_mems = benchmark_tinygrad(self.tiny_model, self.start_tok)

    # note: tokens may differ due to different RoPE implementations or numerical differences
    # but we still report them for debugging (skip check when sampling since it's non-deterministic)
    if TEMPERATURE == 0 and hf_toks != tiny_toks:
      print(f"\nWarning: Generated sequences differ (expected due to numerical differences)")
      print(f"  HF tokens:      {hf_toks[:10]}...")
      print(f"  tinygrad tokens: {tiny_toks[:10]}...")

    table_data = [[i+1, f"{hf_times[i]*1000:.2f}", f"{tiny_times[i]*1000:.2f}", colorize_float(tiny_times[i]/hf_times[i])] for i in range(len(hf_times))]
    print(f"\nBenchmark results ({'sampling' if TEMPERATURE > 0 else 'greedy decoding'}):")
    print(tabulate(table_data, headers=["Token", "HF Torch (ms)", "Tinygrad (ms)", "Ratio"], tablefmt="simple"))

    hf_mean, hf_std = np.mean(hf_times)*1000, np.std(hf_times)*1000
    tiny_mean, tiny_std = np.mean(tiny_times)*1000, np.std(tiny_times)*1000
    print(f"\nAverage times: HF Torch={hf_mean:.2f}±{hf_std:.2f}ms, Tinygrad={tiny_mean:.2f}±{tiny_std:.2f}ms, Ratio={tiny_mean/hf_mean:.2f}")

    # Compute and print Tinygrad memory bandwidth (GB/s) using GlobalCounters measurements
    try:
      # tiny_mems are bytes accessed per token
      tiny_gbps = [(m * 1e-9) / t if t > 0 else 0.0 for m, t in zip(tiny_mems, tiny_times)]
      tiny_gb_mean = np.mean(tiny_gbps)
      tiny_gb_std = np.std(tiny_gbps)
      print(f"Tinygrad memory bandwidth: {tiny_gb_mean:.2f}±{tiny_gb_std:.2f} GB/s (mean±std)")
    except Exception:
      pass

    # Optionally plot token times
    if PLOT:
      try:
        import matplotlib.pyplot as plt
        n = min(len(hf_times), len(tiny_times))
        x = np.arange(1, n+1)
        hf_ms = np.array(hf_times[:n]) * 1000.0
        tiny_ms = np.array(tiny_times[:n]) * 1000.0
        plt.figure()
        plt.plot(x, hf_ms, color='blue', label='HF Torch')
        plt.plot(x, tiny_ms, color='red', label='Tinygrad')
        plt.xlabel('Token')
        plt.ylabel('Time (ms)')
        plt.title(f'Per-token decoding time (BEAM={getenv("BEAM", 0)})')
        plt.legend()
        outfn = 'token_times.png'
        plt.savefig(outfn, dpi=150)
        print(f"Wrote token times plot to {outfn}")
      except Exception as e:
        print(f"Could not plot token times: {e}")

if __name__ == "__main__":
  unittest.main(verbosity=2)
