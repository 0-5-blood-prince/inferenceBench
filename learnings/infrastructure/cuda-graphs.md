# CUDA graphs and allocator settings are a benchmark confound

**Why `cudaFree` blocks.** `cudaMalloc` reserves a virtual address range and
installs GPU page-table mappings. `cudaFree` wants to tear that down, but GPU
execution is asynchronous and the driver does not track per-kernel address
dependencies at fine grain. Its only safe move is to block the calling thread
and drain all in-flight work on the **entire device**, then unmap. That
device-wide drain is why frameworks build userspace caching allocators that
never return blocks to the driver on the hot path.

**Stream-ordered allocation** (`cudaMallocAsync`/`cudaFreeAsync`, CUDA 11.2+)
productises that: frees become stream-ordered rather than host-blocking, memory
returns to a `cudaMemPool_t`, and cross-stream reuse inserts a
`cudaStreamWaitEvent` instead of a host stall. Pools carry a release threshold.
PyTorch exposes the backend via `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`,
though for serving the more useful knob is `expandable_segments:True`, letting a
segment grow in place instead of fragmenting — which matters when KV
allocations vary in size.

**Launch overhead.** Each `cudaLaunchKernel` costs the host a few microseconds.
Irrelevant for a long convolution; fatal for **decode**, where one token is a
full forward pass of hundreds of tiny kernels each running microseconds. Launch
cost becomes comparable to kernel runtime, the CPU cannot refill the queue, and
the GPU idles between kernels — **CPU-launch-bound**, not compute-bound.

**CUDA graphs** collapse hundreds of launches into one `cudaGraphLaunch`. The
catch: **replay bakes in captured pointers and shapes**. Hence bucketed batch
sizes with padding up to the nearest captured bucket; hence a private memory
pool per graph so the caching allocator cannot recycle those addresses; hence
captured graphs consuming memory that competes directly with the KV budget.
Prefill is long, compute-bound and variable, so it is usually left eager.
Current direction is **piecewise capture** of dense static regions with
attention outside, often with `torch.compile`.

## Resolved: measured on both engines, and the asymmetry is structural, not a missing flag

Both were started with their default CUDA-graph settings — no `--cuda-graph-*`
flags passed to either, matching how `run.sh` launches every engine in the
matrix.

**vLLM** captures both prefill-covering and decode graphs:

```
Capturing CUDA graphs (PIECEWISE): 0/51   (piecewise, covers prefill/mixed batches)
Capturing CUDA graphs (FULL):      0/35   (full, decode)
cudagraph_capture_sizes: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, ..., 496, 512]
max_cudagraph_capture_size: 512
```

**SGLang** auto-disables its prefill graph for this model, and caps decode at a
smaller batch:

```
Breakable CUDA graph is incompatible with multimodal model; disabling prefill CUDA graph.
Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled'
Capture target decode CUDA graph begin. backend=full, bs=[1, 2, 4, 8, 12, 16, 24, ..., 248, 256]
```

`cuda_graph_config` from SGLang's `/get_server_info`:
`decode: {backend: full, max_bs: 256, bs: [1,2,4,...,256]}`,
`prefill: {backend: disabled, ...}`.

Two distinct asymmetries, not one:

1. **Prefill graphing.** vLLM graphs prefill-ish batches (piecewise, 51
   buckets); SGLang disables prefill graphing entirely, *automatically*,
   specifically because the model is multimodal — this is not a flag we chose
   or could equalize by passing one, it is SGLang's own safety rule for
   breakable graphs plus multimodal inputs. This is the asymmetry that can
   actually touch TTFT, since prefill is what TTFT measures.
2. **Decode bucket ceiling.** vLLM captures up to batch 512; SGLang caps at
   256. This looks like the more obviously fixable one
   (`--cuda-graph-max-bs-decode`) — but it is very likely immaterial here:
   SGLang's own reported KV capacity is `max_total_num_tokens = 16391` (see
   [../environment.md](environment.md)), which at `T=991` per request ceilings
   live concurrency at roughly 16 requests, and 16 is inside *both* bucket
   sets. We will not generate a batch anywhere near 256, let alone 512, so
   raising SGLang's cap would change nothing observable and was not done.

## Decision, and why it is not "fix before the matrix" after all

SPEC section 6 already states the right policy for exactly this shape of
difference: *"Memory knobs pinned equal across engines; scheduler knobs
recorded, not equalized — the defaults differ and are part of the system under
test."* CUDA graph strategy is the same kind of fact as scheduler policy: an
engine default that is part of what a fair comparison is comparing, not noise
to launder away. Forcing SGLang to graph prefill for a multimodal model is not
offered as a safe option by the engine itself (its own log names the
incompatibility), so "equalize it" is not actually available — only "record it
and read TTFT deltas in light of it" is. That is now done: both engines'
capture behaviour is saved to `gates/<engine>/cuda_graph_info.txt` at every P1
run, per the config-dump step, and the mechanism bullet in the P4 writeup must
cite this fact if Partial-reuse or Cold TTFT gaps show up — a lack of prefill
graphing on SGLang is a plausible independent contributor to any TTFT gap,
separate from caching granularity or scheduling, and needs to be ruled in or
out before attributing a gap to the cache mechanism alone.
