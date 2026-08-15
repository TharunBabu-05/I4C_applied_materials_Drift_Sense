# Dataset V2 Two Noise 100: Three-Way Comparison
**Generated:** August 14, 2026  
**Dataset:** dataset_v2_two_noise_100_14-8-26 (100 pairs)  
**Comparison:** Your Standard vs Your Edge vs Senthil's Fast

---

## 🎯 **OVERALL THREE-WAY COMPARISON**

| Metric | Your Standard | Your Edge | Senthil's Fast | Winner |
|--------|--------------|-----------|--------------|--------|
| **Accuracy** | **95% (95/100)** | 79% (79/100) | 93% (93/100) | **Your Standard** ✅ |
| **Mean Error** | **11.08px** | 39.71px | 17.30px | **Your Standard** ✅ |
| **Median Error** | 0.00px | 0.00px | 0.00px | **TIE** |
| **Max Error** | **313.05px** | 364.09px | 330.00px | **Your Standard** ✅ |
| **Std Error** | **51.59px** | 87.37px | 66.22px | **Your Standard** ✅ |
| **Mean Time** | 150.76ms | 124.06ms | **87.93ms** | **Senthil's** ⚡ |
| **Median Time** | 148.17ms | 121.11ms | 62.58ms | **Senthil's** ⚡ |
| **Total Time** | 15.08s | 12.41s | 8.79s | **Senthil's** ⚡ |

---

## 📊 **DETAILED COMPARISON**

### **Accuracy Performance:**
- **Your Standard:** 95% (95/100) - **Best**
- **Senthil's Fast:** 93% (93/100) - **Second best**
- **Your Edge:** 79% (79/100) - **Worst**

### **Error Performance:**
- **Your Standard:** 11.08px mean error - **Best**
- **Senthil's Fast:** 17.30px mean error - **Second best**
- **Your Edge:** 39.71px mean error - **Worst**

### **Speed Performance:**
- **Senthil's Fast:** 87.93ms mean - **Fastest (1.7x vs your standard)**
- **Your Edge:** 124.06ms mean - **Medium speed**
- **Your Standard:** 150.76ms mean - **Slowest**

---

## 🔍 **MODE-SPECIFIC FAILURE ANALYSIS**

### **Your Standard Mode (5 failures):**
| Pair | Ground Truth | Predicted | Error | Degradation |
|------|-------------|-----------|-------|--------------|
| **pair_014** | (310, 500) | (310, 785) | 285.0px | gaussian_blur, field_nonuniformity |
| **pair_048** | (500, 310) | (668, 454) | 221.3px | gaussian_blur, field_nonuniformity |
| **pair_061** | (650, 350) | (576, 348) | 74.0px | gaussian_blur, field_nonuniformity |
| **pair_064** | (350, 630) | (490, 910) | 313.0px | gaussian_blur, field_nonuniformity |
| **pair_073** | (500, 690) | (689, 785) | 211.5px | gaussian_blur, field_nonuniformity |

### **Senthil's Fast Mode (7 failures):**
| Pair | Ground Truth | Predicted | Error | Degradation | vs Your Standard |
|------|-------------|-----------|-------|--------------|------------------|
| **pair_011** | (500, 500) | (500, 170) | 330.0px | gaussian_blur, field_nonuniformity | New failure |
| **pair_014** | (310, 500) | (120, 690) | 268.7px | gaussian_blur, field_nonuniformity | Same failure |
| **pair_015** | (500, 405) | (500, 690) | 285.0px | gaussian_blur, field_nonuniformity | New failure |
| **pair_061** | (650, 350) | (576, 348) | 74.0px | gaussian_blur, field_nonuniformity | Same failure |
| **pair_064** | (350, 630) | (420, 910) | 288.6px | gaussian_blur, field_nonuniformity | Same failure |
| **pair_073** | (500, 690) | (310, 880) | 268.7px | gaussian_blur, field_nonuniformity | Same failure |
| **pair_075** | (501, 405) | (311, 500) | 212.4px | gaussian_blur, field_nonuniformity | New failure |

**Key Observation:** Senthil's mode has 2 additional failures compared to your standard mode, but also has 2 fewer failures compared to your edge mode.

---

## 📈 **PERFORMANCE BREAKDOWN**

### **Accuracy vs. Speed Trade-off:**

**Your Standard Mode:**
- ✅ **Best accuracy:** 95%
- ✅ **Best error metrics:** 11.08px mean error
- ❌ **Slowest:** 150.76ms per pair
- **Trade-off:** Prioritizes accuracy over speed

**Senthil's Fast Mode:**
- ✅ **Excellent accuracy:** 93% (only 2% worse than your standard)
- ✅ **Good error metrics:** 17.30px mean error (56% higher than yours)
- ⚡ **Fastest:** 87.93ms per pair (1.7x speedup vs your standard)
- **Trade-off:** Good balance of accuracy and speed

**Your Edge Enhancement Mode:**
- ❌ **Poor accuracy:** 79% (16% worse than your standard)
- ❌ **Worst error metrics:** 39.71px mean error
- ⚡ **Medium speed:** 124.06ms per pair
- **Trade-off:** Speed gain not worth accuracy loss

---

## 🔬 **TECHNICAL ANALYSIS**

### **Why Senthil's Fast Mode Performs Well:**

**Strengths:**
1. ✅ **ROI-guided search:** Efficient 320x320 windows around top-10 candidates
2. ✅ **OpenCV optimization:** Explicit SIMD, contiguous arrays
3. ✅ **Memory efficiency:** uint8 throughout
4. ✅ **Fast image loading:** cv2.imread() vs PIL
5. ✅ **Reduced candidate tracking:** Top-15 → Top-10 → Top-3

**Why it almost matches your accuracy:**
- **ROI approach:** Works well for most DRAM patterns
- **SIMD optimization:** Better CPU utilization
- **Contiguous memory:** Better cache performance
- **Smart candidate reduction:** Maintains most important candidates

### **Why Your Standard Mode Still Wins:**

**Advantages:**
1. ✅ **Full search space:** No ROI limitations that could exclude true location
2. ✅ **Center-bias disambiguation:** Handles periodic ambiguity better
3. ✅ **Better preprocessing:** Tuned for DRAM-specific noise patterns
4. ✅ **More candidate tracking:** Maintains 20-30 candidates vs 10-3
5. ✅ **Proven parameters:** Optimized through extensive testing

**Why it wins:**
- **2% accuracy advantage:** 95% vs 93%
- **Better error metrics:** 11.08px vs 17.30px mean error
- **More consistent:** Lower std error (51.59px vs 66.22px)
- **Robustness:** Handles edge cases better

### **Why Your Edge Mode Fails:**

**Problems:**
1. ❌ **Noise amplification:** Sobel edge detection amplifies SEM shot noise
2. ❌ **False edges:** Creates spurious correlations in noisy regions
3. ❌ **Edge/intensity blend:** 60% edge + 40% intensity not optimal for this dataset
4. ❌ **Degradation sensitivity:** More sensitive to gaussian blur + field nonuniformity
5. ❌ **Periodic confusion:** Edge enhancement confuses periodic patterns

---

## 🎯 **PRACTICAL IMPLICATIONS**

### **For Competition Use:**

**Your Standard Mode:**
- ✅ **Best accuracy:** 95% (highest among all three)
- ✅ **Best error metrics:** 11.08px mean error
- ✅ **Most consistent:** Lowest std error
- ✅ **Competition priority:** Accuracy is paramount
- ❌ **Slowest:** 150.76ms per pair (but still acceptable)

**Senthil's Fast Mode:**
- ⚠️ **Excellent alternative:** 93% accuracy (only 2% worse)
- ⚡ **Fastest speed:** 87.93ms per pair (1.7x speedup)
- ⚠️ **Good balance:** Accuracy/speed trade-off is reasonable
- ⚠️ **ROI limitations:** May fail on edge cases

**Your Edge Mode:**
- ❌ **Not recommended:** 79% accuracy (16% worse)
- ❌ **Worst error metrics:** 39.71px mean error
- ❌ **Degraded performance:** Edge enhancement hurts DRAM patterns

### **Speed vs. Accuracy Analysis:**

**Senthil's Speed Advantage:**
- **1.7x faster:** 87.93ms vs 150.76ms
- **Speed gain:** 42% reduction in processing time
- **Accuracy cost:** 2% loss (95% → 93%)
- **Trade-off:** Speed gain worth minimal accuracy loss

**Your Edge Mode Trade-off:**
- **1.2x faster:** 124.06ms vs 150.76ms
- **Speed gain:** 18% reduction in processing time
- **Accuracy cost:** 16% loss (95% → 79%)
- **Trade-off:** Speed gain NOT worth significant accuracy loss

---

## 🏁 **FINAL VERDICT**

### **For I4C Hackathon Competition:**
**Your standard mode remains the best choice:**
- ✅ **Highest accuracy:** 95% (vs 93% for Senthil's, 79% for edge)
- ✅ **Best error metrics:** 11.08px mean error (vs 17.30px for Senthil's)
- ✅ **Most consistent:** Lowest std error (51.59px vs 66.22px)
- ✅ **Competition priority:** Accuracy is more important than speed
- ✅ **Acceptable speed:** 150.76ms is fast enough (6.6 FPS)

### **When Senthil's Fast Mode Could Be Considered:**
- ⚡ **High-throughput scenarios:** If speed becomes critical
- ⚡ **Production deployment:** 1.7x speedup could be valuable
- ⚠️ **Acceptable accuracy:** 93% is still excellent
- ⚠️ **Risk tolerance:** 2% accuracy loss must be acceptable

### **Recommendation:**
**Submit your standard mode for the I4C hackathon competition.** The 2% accuracy advantage over Senthil's fast mode, combined with better error metrics and consistency, makes it the optimal choice for competition where accuracy is the primary evaluation metric.

**Senthil's fast mode is an excellent alternative for production/high-throughput scenarios, but for competition where accuracy is paramount, your standard mode is superior.**

---

## 📋 **COMPARISON SUMMARY**

### **Overall Rankings:**

**Accuracy:** Your Standard (95%) > Senthil's Fast (93%) > Your Edge (79%)
**Error Metrics:** Your Standard (11.08px) > Senthil's Fast (17.30px) > Your Edge (39.71px)
**Speed:** Senthil's Fast (87.93ms) > Your Edge (124.06ms) > Your Standard (150.76ms)
**Consistency:** Your Standard (51.59px std) > Senthil's Fast (66.22px std) > Your Edge (87.37px std)

### **Final Assessment:**
**Your standard mode achieves the best balance of accuracy, error metrics, and consistency for the I4C hackathon competition. While Senthil's fast mode offers a compelling speed advantage with minimal accuracy loss, the competition's focus on accuracy makes your standard mode the optimal choice.**

---

## 🎉 **CONCLUSION**

**Three-way comparison shows:**
- ✅ **Your standard mode:** 95% accuracy, 11.08px mean error (best overall)
- ⚡ **Senthil's fast mode:** 93% accuracy, 87.93ms mean time (best speed trade-off)
- ❌ **Your edge mode:** 79% accuracy, 39.71px mean error (not recommended)

**Final recommendation:** Continue with your standard mode for the I4C hackathon competition. It achieves the highest accuracy and best error metrics, which are the primary evaluation criteria for this competition.