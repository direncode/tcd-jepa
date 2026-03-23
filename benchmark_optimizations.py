#!/usr/bin/env python3
"""
Comprehensive benchmark suite measuring optimization gains in TCD-JEPA.

Tests:
1. Flash Attention (SDPA) vs Manual Attention - latency & memory
2. Hook overhead - disabled vs enabled statistics collection
3. Throughput scaling - images/sec across batch sizes
4. Full training step - forward + backward + EMA update
5. Memory footprint - peak allocation tracking
6. Batch scaling efficiency
"""

import time
import gc
import sys
import statistics
import torch
import torch.nn as nn

from tcd_jepa.models.vision_transformer import Attention, Block, VisionTransformer
from tcd_jepa.models.context_encoder import ContextEncoder
from tcd_jepa.models.tcd_jepa_model import build_tcd_jepa, TCDJEPAModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # use float32 for fair CPU comparison; bf16 tested separately
WARMUP = 3
REPEATS = 20

def fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds*1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"

def fmt_mem(bytes_val):
    if bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f} MB"
    else:
        return f"{bytes_val/1024**3:.2f} GB"

def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

def reset_memory():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

def get_peak_memory():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated()
    return 0

def benchmark_fn(fn, warmup=WARMUP, repeats=REPEATS):
    """Benchmark a function, return (mean_time, std_time, peak_memory)."""
    # Warmup
    for _ in range(warmup):
        fn()
        sync()

    reset_memory()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = get_peak_memory()
    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0
    return mean_t, std_t, peak_mem


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_comparison(name_a, time_a, mem_a, name_b, time_b, mem_b):
    speedup = time_a / time_b if time_b > 0 else float('inf')
    mem_saved = (1 - mem_b / mem_a) * 100 if mem_a > 0 else 0
    print(f"  {name_a:30s}: {fmt_time(time_a):>12s}  mem: {fmt_mem(mem_a):>10s}")
    print(f"  {name_b:30s}: {fmt_time(time_b):>12s}  mem: {fmt_mem(mem_b):>10s}")
    print(f"  {'Speedup':30s}: {speedup:.2f}x")
    if mem_a > 0:
        print(f"  {'Memory saved':30s}: {mem_saved:.1f}%")


# ============================================================
# TEST 1: Flash Attention (SDPA) vs Manual Attention
# ============================================================
def test_attention_sdpa_vs_manual():
    print_header("TEST 1: SDPA (Flash Attention) vs Manual Attention")

    configs = [
        ("Small  (dim=192, heads=3, seq=64)", 192, 3, 64),
        ("Base   (dim=384, heads=6, seq=196)", 384, 6, 196),
        ("Large  (dim=768, heads=12, seq=196)", 768, 12, 196),
        ("XLarge (dim=1024, heads=16, seq=196)", 1024, 16, 196),
    ]

    batch_size = 32

    for name, dim, heads, seq_len in configs:
        print(f"\n  Config: {name}, batch={batch_size}")

        attn = Attention(dim=dim, num_heads=heads, qkv_bias=True).to(DEVICE)
        attn.eval()
        x = torch.randn(batch_size, seq_len, dim, device=DEVICE)

        # SDPA path (optimized - default)
        def run_sdpa():
            with torch.no_grad():
                attn(x, return_attention=False)

        # Manual path (fallback)
        def run_manual():
            with torch.no_grad():
                attn(x, return_attention=True)

        t_sdpa, std_sdpa, mem_sdpa = benchmark_fn(run_sdpa)
        reset_memory()
        t_manual, std_manual, mem_manual = benchmark_fn(run_manual)

        print_comparison("Manual attention", t_manual, mem_manual,
                        "SDPA (Flash Attention)", t_sdpa, mem_sdpa)

        del attn, x
        reset_memory()


# ============================================================
# TEST 2: Hook Overhead (Stats Collection)
# ============================================================
def test_hook_overhead():
    print_header("TEST 2: Hook Overhead - Stats Collection Disabled vs Enabled")

    configs = [
        ("Small (dim=192, depth=6)", 32, 4, 192, 6, 3),
        ("Base  (dim=384, depth=12)", 32, 4, 384, 12, 6),
    ]

    batch_size = 32

    for name, img_size, patch_size, embed_dim, depth, num_heads in configs:
        print(f"\n  Config: {name}, batch={batch_size}")

        # Build encoder WITHOUT hooks (optimized)
        vit_no = VisionTransformer(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )
        enc_no_hooks = ContextEncoder(vit_no, collect_stats=False).to(DEVICE)
        enc_no_hooks.eval()

        # Build encoder WITH hooks (unoptimized)
        vit_hooks = VisionTransformer(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )
        enc_hooks = ContextEncoder(vit_hooks, collect_stats=True).to(DEVICE)
        enc_hooks.eval()

        num_patches = (img_size // patch_size) ** 2
        images = torch.randn(batch_size, 3, img_size, img_size, device=DEVICE)
        masks = [torch.arange(num_patches // 2, device=DEVICE).unsqueeze(0).expand(batch_size, -1)]

        def run_no_hooks():
            with torch.no_grad():
                enc_no_hooks(images, masks)

        def run_hooks():
            with torch.no_grad():
                enc_hooks(images, masks)

        t_hooks, _, mem_hooks = benchmark_fn(run_hooks)
        reset_memory()
        t_no_hooks, _, mem_no_hooks = benchmark_fn(run_no_hooks)

        print_comparison("With hooks (stats ON)", t_hooks, mem_hooks,
                        "No hooks (stats OFF)", t_no_hooks, mem_no_hooks)

        print(f"  {'Hooks registered':30s}: {len(enc_hooks._hooks)} vs {len(enc_no_hooks._hooks)}")

        del enc_no_hooks, enc_hooks, images
        reset_memory()


# ============================================================
# TEST 3: Throughput - Images/sec
# ============================================================
def test_throughput():
    print_header("TEST 3: Forward Pass Throughput (images/sec)")

    configs = [
        ("Small", 32, 4, 192, 6, 3, 96, 2, 3),
        ("Base", 32, 4, 384, 12, 6, 192, 4, 6),
        ("Large", 32, 4, 768, 12, 12, 384, 4, 12),
    ]

    batch_sizes = [16, 32, 64, 128, 256]

    for name, img_size, patch_size, embed_dim, depth, num_heads, pred_dim, pred_depth, pred_heads in configs:
        print(f"\n  Model: {name} (embed={embed_dim}, depth={depth})")

        model = build_tcd_jepa(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            predictor_embed_dim=pred_dim, predictor_depth=pred_depth,
            predictor_num_heads=pred_heads
        ).to(DEVICE)
        model.eval()

        num_patches = (img_size // patch_size) ** 2
        half = num_patches // 2

        print(f"  {'Batch':>8s} | {'Time':>10s} | {'Img/sec':>10s} | {'Peak Mem':>10s}")
        print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10}")

        for bs in batch_sizes:
            try:
                images = torch.randn(bs, 3, img_size, img_size, device=DEVICE)
                masks_enc = [torch.arange(half, device=DEVICE).unsqueeze(0).expand(bs, -1)]
                masks_pred = [torch.arange(half, num_patches, device=DEVICE).unsqueeze(0).expand(bs, -1)]

                def run():
                    with torch.no_grad():
                        model(images, masks_enc, masks_pred)

                t, std, mem = benchmark_fn(run, warmup=2, repeats=10)
                ips = bs / t
                print(f"  {bs:>8d} | {fmt_time(t):>10s} | {ips:>10.1f} | {fmt_mem(mem):>10s}")

                del images
                reset_memory()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  {bs:>8d} | {'OOM':>10s} | {'---':>10s} | {'---':>10s}")
                    reset_memory()
                    break
                raise

        del model
        reset_memory()


# ============================================================
# TEST 4: Full Training Step (Forward + Backward + EMA)
# ============================================================
def test_training_step():
    print_header("TEST 4: Full Training Step (Forward + Backward + EMA Update)")

    configs = [
        ("Small", 32, 4, 192, 6, 3, 96, 2, 3),
        ("Base", 32, 4, 384, 12, 6, 192, 4, 6),
    ]

    batch_size = 64

    for name, img_size, patch_size, embed_dim, depth, num_heads, pred_dim, pred_depth, pred_heads in configs:
        print(f"\n  Model: {name}, batch={batch_size}")

        model = build_tcd_jepa(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            predictor_embed_dim=pred_dim, predictor_depth=pred_depth,
            predictor_num_heads=pred_heads
        ).to(DEVICE)
        model.train()

        # Setup optimizer (like the real training)
        params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)

        num_patches = (img_size // patch_size) ** 2
        half = num_patches // 2

        images = torch.randn(batch_size, 3, img_size, img_size, device=DEVICE)
        masks_enc = [torch.arange(half, device=DEVICE).unsqueeze(0).expand(batch_size, -1)]
        masks_pred = [torch.arange(half, num_patches, device=DEVICE).unsqueeze(0).expand(batch_size, -1)]

        momentum = 0.996

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            result = model(images, masks_enc, masks_pred)
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            # EMA update
            with torch.no_grad():
                for p_ctx, p_tgt in zip(model.context_encoder.parameters(),
                                         model.target_encoder.parameters()):
                    p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1 - momentum)

        t, std, mem = benchmark_fn(train_step, warmup=3, repeats=15)
        ips = batch_size / t

        param_count = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in params)

        print(f"  {'Total params':30s}: {param_count:,}")
        print(f"  {'Trainable params':30s}: {trainable:,}")
        print(f"  {'Step time':30s}: {fmt_time(t)} (±{fmt_time(std)})")
        print(f"  {'Throughput':30s}: {ips:.1f} img/sec")
        print(f"  {'Peak memory':30s}: {fmt_mem(mem)}")
        print(f"  {'Steps/min (projected)':30s}: {60/t:.0f}")

        del model, optimizer, images
        reset_memory()


# ============================================================
# TEST 5: Mixed Precision (bf16) Gains
# ============================================================
def test_mixed_precision():
    print_header("TEST 5: Mixed Precision (bfloat16) vs float32")

    if DEVICE != "cuda":
        print("  Skipped: CUDA not available")
        return

    if not torch.cuda.is_bf16_supported():
        print("  Skipped: bf16 not supported on this device")
        return

    model = build_tcd_jepa(
        img_size=32, patch_size=4,
        embed_dim=384, depth=12, num_heads=6,
        predictor_embed_dim=192, predictor_depth=4, predictor_num_heads=6
    ).to(DEVICE)
    model.train()

    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    batch_size = 64
    num_patches = 64
    half = 32
    images = torch.randn(batch_size, 3, 32, 32, device=DEVICE)
    masks_enc = [torch.arange(half, device=DEVICE).unsqueeze(0).expand(batch_size, -1)]
    masks_pred = [torch.arange(half, num_patches, device=DEVICE).unsqueeze(0).expand(batch_size, -1)]

    def step_fp32():
        optimizer.zero_grad(set_to_none=True)
        result = model(images, masks_enc, masks_pred)
        result["loss"].backward()
        optimizer.step()

    def step_bf16():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model(images, masks_enc, masks_pred)
        result["loss"].backward()
        optimizer.step()

    t_fp32, _, mem_fp32 = benchmark_fn(step_fp32, warmup=3, repeats=10)
    reset_memory()
    t_bf16, _, mem_bf16 = benchmark_fn(step_bf16, warmup=3, repeats=10)

    print_comparison("float32", t_fp32, mem_fp32, "bfloat16", t_bf16, mem_bf16)

    del model, optimizer, images
    reset_memory()


# ============================================================
# TEST 6: Batch Scaling Efficiency
# ============================================================
def test_batch_scaling():
    print_header("TEST 6: Batch Scaling Efficiency (Linear Scaling Test)")

    model = build_tcd_jepa(
        img_size=32, patch_size=4,
        embed_dim=384, depth=12, num_heads=6,
        predictor_embed_dim=192, predictor_depth=4, predictor_num_heads=6
    ).to(DEVICE)
    model.eval()

    num_patches = 64
    half = 32
    batch_sizes = [8, 16, 32, 64, 128]
    results = []

    print(f"  {'Batch':>8s} | {'Time':>10s} | {'Img/sec':>10s} | {'Per-img':>10s} | {'Efficiency':>10s}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    base_per_img = None

    for bs in batch_sizes:
        try:
            images = torch.randn(bs, 3, 32, 32, device=DEVICE)
            masks_enc = [torch.arange(half, device=DEVICE).unsqueeze(0).expand(bs, -1)]
            masks_pred = [torch.arange(half, num_patches, device=DEVICE).unsqueeze(0).expand(bs, -1)]

            def run():
                with torch.no_grad():
                    model(images, masks_enc, masks_pred)

            t, _, mem = benchmark_fn(run, warmup=2, repeats=10)
            ips = bs / t
            per_img = t / bs

            if base_per_img is None:
                base_per_img = per_img

            efficiency = (base_per_img / per_img) * 100
            results.append((bs, t, ips, per_img, efficiency))

            print(f"  {bs:>8d} | {fmt_time(t):>10s} | {ips:>10.1f} | {fmt_time(per_img):>10s} | {efficiency:>9.1f}%")

            del images
            reset_memory()
        except RuntimeError:
            break

    del model
    reset_memory()


# ============================================================
# TEST 7: Component-level Breakdown
# ============================================================
def test_component_breakdown():
    print_header("TEST 7: Component-Level Latency Breakdown")

    model = build_tcd_jepa(
        img_size=32, patch_size=4,
        embed_dim=384, depth=12, num_heads=6,
        predictor_embed_dim=192, predictor_depth=4, predictor_num_heads=6
    ).to(DEVICE)
    model.eval()

    batch_size = 64
    num_patches = 64
    half = 32
    images = torch.randn(batch_size, 3, 32, 32, device=DEVICE)
    masks_enc = [torch.arange(half, device=DEVICE).unsqueeze(0).expand(batch_size, -1)]
    masks_pred = [torch.arange(half, num_patches, device=DEVICE).unsqueeze(0).expand(batch_size, -1)]

    # Patch embedding
    def run_patch_embed():
        with torch.no_grad():
            x = model.context_encoder.encoder.patch_embed(images)

    # Context encoder full
    def run_context_enc():
        with torch.no_grad():
            model.context_encoder(images, masks_enc)

    # Target encoder full
    def run_target_enc():
        with torch.no_grad():
            model.target_encoder(images, masks_pred)

    # Full forward
    def run_full():
        with torch.no_grad():
            model(images, masks_enc, masks_pred)

    components = [
        ("Patch Embedding", run_patch_embed),
        ("Context Encoder", run_context_enc),
        ("Target Encoder", run_target_enc),
        ("Full Forward", run_full),
    ]

    total_time = 0
    times = {}
    for comp_name, fn in components:
        reset_memory()
        t, std, mem = benchmark_fn(fn, warmup=2, repeats=10)
        times[comp_name] = t
        print(f"  {comp_name:30s}: {fmt_time(t):>10s} (±{fmt_time(std)})  mem: {fmt_mem(mem):>10s}")

    full_t = times["Full Forward"]
    print(f"\n  Component share of full forward:")
    for comp_name in ["Patch Embedding", "Context Encoder", "Target Encoder"]:
        pct = times[comp_name] / full_t * 100
        print(f"    {comp_name:28s}: {pct:5.1f}%")

    predictor_t = full_t - times["Context Encoder"] - times["Target Encoder"]
    print(f"    {'Predictor + Loss (estimated)':28s}: {predictor_t/full_t*100:5.1f}%")

    del model, images
    reset_memory()


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"{'='*70}")
    print(f"  TCD-JEPA Optimization Benchmark Suite")
    print(f"{'='*70}")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  CUDA: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(0).total_mem
        print(f"  GPU Memory: {fmt_mem(total_mem)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  SDPA available: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")
    print(f"  Warmup: {WARMUP}, Repeats: {REPEATS}")

    test_attention_sdpa_vs_manual()
    test_hook_overhead()
    test_throughput()
    test_training_step()
    test_mixed_precision()
    test_batch_scaling()
    test_component_breakdown()

    print_header("BENCHMARK COMPLETE")
    print("  All tests finished successfully.\n")


if __name__ == "__main__":
    main()
