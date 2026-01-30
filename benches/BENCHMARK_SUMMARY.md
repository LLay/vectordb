# Benchmark Summary

## Quick Reference

### For Daily Development (< 10 seconds)
```bash
cargo run --release --example quick_recall_check
```
**Perfect for:** Quick sanity checks after code changes

---

### For Fast Iteration (< 2 minutes)
```bash
cargo bench --bench speed_fast    # ~70s - latency only
cargo bench --bench recall_fast   # ~30s - recall + latency
```
**Perfect for:** Testing parameter changes, comparing configurations

---

### For Production Validation (2-10 minutes)
```bash
cargo bench --bench recall_proper      # ~3m - comprehensive recall
cargo bench --bench tune_tree_params   # ~5m - find optimal params
cargo bench --bench profile_bench      # ~10m - full profiling with 1M vectors
```
**Perfect for:** Final validation before deployment

---

## All Available Benchmarks

| Name | Type | Time | Dataset | Purpose |
|------|------|------|---------|---------|
| `quick_recall_check` | Example | 8s | 1K×128d | Quick sanity check |
| `speed_fast` | Bench | 70s | 5K×128d | Fast latency measurement |
| `recall_fast` | Bench | 30s | 2K×128d | Fast recall testing |
| `recall_proper` | Bench | 3m | 10K×256d | Comprehensive recall |
| `tune_tree_params` | Bench | 5m | 10K×256d | Parameter optimization |
| `profile_bench` | Bench | 10m | 1M×1024d | Production profiling |
| `recall_bench` | Bench | 5m | 50K×512d | Original recall benchmark |

---

## Typical Workflow

### 1. Active Development
```bash
# After every change
cargo run --release --example quick_recall_check  # 8s
```

### 2. Before Committing
```bash
# Verify no regressions
cargo bench --bench speed_fast    # 70s
cargo test --release              # 10s
```

### 3. Before Production
```bash
# Full validation
cargo bench --bench recall_proper      # 3m
cargo bench --bench profile_bench      # 10m
```

---

## Example Outputs

### quick_recall_check (8 seconds)
```
Config          Recall@10    Latency(μs)  Build(ms)
-------------------------------------------------------
default         15.0%        45.2         5
tuned           23.0%        108.7        7
high_recall     25.0%        133.4        9
```

### speed_fast (70 seconds)
```
speed_fast/low_latency     time: [21.7 μs 21.7 μs 21.8 μs]
speed_fast/balanced        time: [90.4 μs 91.6 μs 92.9 μs]
speed_fast/high_recall     time: [104.1 μs 104.6 μs 105.1 μs]
speed_fast/thorough        time: [108.8 μs 110.2 μs 112.3 μs]
```

### recall_proper (3 minutes)
```
--- Scenario: In-Dataset ---
p1_r2          11.5%        27.2μs
p2_r3          13.0%        108.1μs
p3_r5          13.6%        119.3μs

--- Scenario: Random ---
p1_r2          1.0%         27.2μs
p5_r5          5.1%         160.7μs
```

---

## Tips

1. **Use `quick_recall_check` most often** - it's the fastest feedback loop
2. **Save baselines to track progress:**
   ```bash
   cargo bench --bench speed_fast -- --save-baseline v1
   cargo bench --bench speed_fast -- --baseline v1
   ```
3. **Filter specific tests:**
   ```bash
   cargo bench --bench speed_fast -- balanced
   ```
4. **Run in release mode always** - debug builds are 10-100x slower

---

## Understanding Results

### Recall
- **<10%**: Too low, increase `probes` or reduce `max_leaf_size`
- **10-30%**: Low, acceptable for speed-critical applications
- **30-60%**: Moderate, good balance
- **60-80%**: Good, suitable for most applications
- **>80%**: Excellent, near-optimal

### Latency (for 10K vectors)
- **<50μs**: Excellent - low latency config
- **50-100μs**: Good - balanced config
- **100-200μs**: Acceptable - high recall config
- **>200μs**: Slow - too many probes or large leaves

### For 1M vectors, expect ~3-5x higher latency

---

## Current Performance (as of latest run)

### Speed (5K vectors, 128d)
- Low latency: **21.7μs**
- Balanced: **91.6μs**
- High recall: **104.6μs**

### Recall (1K vectors, in-dataset)
- Default (p=2,r=3): **15%**
- Tuned (p=3,r=3): **23%**
- High recall (p=5,r=3): **25%**

**Note:** Low recall is due to default parameters. See `RECALL_ANALYSIS.md` for tuning guidance.

---

## See Also

- `RECALL_ANALYSIS.md` - Understanding and improving recall
- `QUICK_BENCHMARKS.md` - Detailed benchmark documentation
- `BENCHMARKS.md` - Original benchmark overview
