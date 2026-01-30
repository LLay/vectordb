# Benchmark Quick Start

## Measure Your Current Performance

### Option 1: Super Quick (30 seconds)
```bash
cargo run --release --example profile_query
```

**You'll see:**
```
Configuration: Balanced (probes=2)
Query Latency:
  Mean:   3.245 ms
  Median: 3.120 ms
  p99:    6.180 ms
```

### Option 2: Full Benchmark (5-10 minutes)
```bash
cargo bench --bench profile_bench
```

**You'll get:**
- Detailed criterion statistics
- Confidence intervals
- HTML reports with charts
- Saved for future comparison

---

## Track Optimization Progress

### Step 1: Save Baseline (Before Optimizing)
```bash
cargo bench --bench profile_bench -- --save-baseline v1_baseline
```

### Step 2: Make Your Changes
```rust
// Implement binary quantization, parallel scanning, etc.
// Follow SPEED_CHECKLIST.md
```

### Step 3: Compare Results
```bash
cargo bench --bench profile_bench -- --baseline v1_baseline
```

**Criterion will show:**
```
query_profile_1M/balanced
  change: [-32.567%]  ← 32% FASTER! 🎉
  Performance has improved.
```

---

## What Each Benchmark Tells You

### `profile_bench.rs` - The Main Profiler

**What it measures:**

1. **query_profile_1M** - Your main target
   - Tests with 1M vectors
   - Different configs (low latency, balanced, high recall)
   - This is your optimization target!

2. **query_profile_varying_size** - Scaling behavior
   - Tests 10K, 100K, 1M vectors
   - Shows logarithmic vs linear scaling
   - Validates tree depth works correctly

3. **tree_depth_impact** - Parameter tuning
   - Tests different branching factors
   - Helps find optimal tree structure
   - Run this to tune your index

4. **cache_simulation** - Cache behavior
   - Repeated vs random queries
   - Shows hot vs cold performance
   - Validates caching strategy

5. **parallelism_potential** - Batch performance
   - Sequential vs parallel batch queries
   - Measures throughput
   - Important for production

### `profile_query.rs` - Quick Iteration Tool

**Use this when:**
- Making small changes
- Want fast feedback
- Don't need statistical rigor
- Just checking if it got faster

**Don't use this when:**
- Publishing results
- Detecting regressions
- Need confidence intervals
- Want historical tracking

---

## Benchmark Workflow Examples

### Example 1: First Time Setup
```bash
# 1. See current performance
cargo run --release --example profile_query

# 2. Save baseline for comparison
cargo bench --bench profile_bench -- --save-baseline initial

# 3. View HTML report
open target/criterion/report/index.html
```

### Example 2: After Each Optimization
```bash
# Quick check
cargo run --release --example profile_query
# Look at median - did it improve?

# If yes, run full benchmark
cargo bench --bench profile_bench -- --baseline initial

# Save new baseline
cargo bench --bench profile_bench -- --save-baseline v2_optimized
```

### Example 3: Comparing Multiple Versions
```bash
# You have: v1_baseline, v2_quantized, v3_parallel

# Compare v1 vs v3 (total improvement)
cargo bench --bench profile_bench -- --baseline v1_baseline

# Compare v2 vs v3 (impact of last change)  
cargo bench --bench profile_bench -- --baseline v2_quantized

# This shows incremental improvements!
```

---

## Reading Criterion Output

### Basic Output
```
query_profile_1M/balanced
                        time:   [2.1234 ms 2.1890 ms 2.2654 ms]
                        thrpt:  [456.78 elem/s 457.12 elem/s 457.45 elem/s]
```

**Interpretation:**
- `time: [low, estimate, high]` - 95% confidence interval
- `thrpt:` - Throughput (elements per second)
- Estimate (middle value) is most reliable

### With Baseline Comparison
```
query_profile_1M/balanced
                        time:   [2.1234 ms 2.1890 ms 2.2654 ms]
                        change: [-35.234% -32.567% -29.891%] (p = 0.00 < 0.05)
                        Performance has improved.
```

**Interpretation:**
- `change: [-35% -32% -29%]` - You're 32% faster! 🚀
- `p = 0.00 < 0.05` - Statistically significant (not random)
- "Performance has improved" - Regression detection

### Warning Signs
```
query_profile_1M/balanced
                        time:   [3.5 ms 3.7 ms 4.2 ms]
                        change: [+15% +23% +35%] (p = 0.02 < 0.05)
                        Performance has regressed.
```

**Interpretation:**
- Positive change = SLOWER ❌
- Your optimization made things worse!
- Revert and try different approach

---

## Common Issues

### Issue: High Variance
```
Found 15 outliers among 100 measurements (15.00%)
  10 (10.00%) high mild
  5 (5.00%) high severe
```

**Solutions:**
1. Close other applications
2. Run on battery (consistent CPU freq)
3. Increase sample size
4. Disable CPU frequency scaling

### Issue: "No baseline found"
```
Error: Baseline 'before_opt' not found
```

**Solution:**
```bash
# Create it first
cargo bench -- --save-baseline before_opt
```

### Issue: Benchmarks too slow
```bash
# Quick benchmarks (less accurate)
cargo bench -- --quick

# Or increase sample size in code
group.sample_size(10); // Faster
```

---

## Best Practices

### ✅ Do:
- Save baseline before optimizing
- Run quick profiler first (fast feedback)
- Use full benchmark for final validation
- Check HTML reports (great visualizations)
- Run on battery power (consistent)
- Close other apps
- Warm up caches before measuring

### ❌ Don't:
- Trust single runs (use criterion)
- Skip baselines (can't track progress)
- Run with other apps open
- Ignore statistical significance
- Compare different hardware

---

## Expected Improvements

Track your progress against these targets:

| Optimization | Expected Improvement | Target Latency |
|-------------|---------------------|----------------|
| Baseline | - | 3-6ms (p99) |
| + Binary quantization | 30-40% faster | 2-4ms |
| + Parallel scanning | 50% faster | 1-2ms |
| + mmap + caching | 20% faster | 1-2ms |
| **Final (laptop)** | **3-5x total** | **1-2ms p99** ✅ |

---

## Quick Commands Reference

| Command | Purpose | Time |
|---------|---------|------|
| `cargo run --release --example profile_query` | Quick check | 30s |
| `cargo bench --bench profile_bench -- --save-baseline NAME` | Save baseline | 5-10min |
| `cargo bench --bench profile_bench -- --baseline NAME` | Compare to baseline | 5-10min |
| `cargo bench --bench profile_bench -- query_profile_1M` | Just main benchmark | 2-3min |
| `open target/criterion/report/index.html` | View reports | - |

---

## TL;DR

**To measure progress:**

1. **First time:**
   ```bash
   cargo bench --bench profile_bench -- --save-baseline before
   ```

2. **After each change:**
   ```bash
   cargo run --release --example profile_query  # Quick check
   cargo bench --bench profile_bench -- --baseline before  # Full validation
   ```

3. **View results:**
   ```bash
   open target/criterion/report/index.html
   ```

**Start benchmarking now!** 📊
