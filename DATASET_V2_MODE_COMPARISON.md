# Dataset V2 Two Noise 100: Standard vs Edge Enhancement Mode Comparison
**Generated:** August 14, 2026  
**Dataset:** dataset_v2_two_noise_100_14-8-26 (100 pairs)  
**Comparison:** Standard Mode vs Edge Enhancement Mode

---

## 🎯 **OVERALL COMPARISON**

| Metric | Standard Mode | Edge Enhancement Mode | Winner |
|--------|--------------|----------------------|--------|
| **Accuracy** | **95% (95/100)** | 79% (79/100) | **Standard** ✅ |
| **Mean Error** | **11.08px** | 39.71px | **Standard** ✅ |
| **Median Error** | 0.00px | 0.00px | **TIE** |
| **Max Error** | **313.05px** | 364.09px | **Standard** ✅ |
| **Min Error** | 0.00px | 0.00px | **TIE** |
| **Std Error** | **51.59px** | 87.37px | **Standard** ✅ |
| **Mean Time** | 150.76ms | **124.06ms** | **Edge** ⚡ |
| **Median Time** | 148.17ms | 121.11ms | **Edge** ⚡ |

---

## 📊 **DETAILED PERFORMANCE COMPARISON**

### **Accuracy Comparison:**
- **Standard Mode:** 95% accuracy (95/100 pairs)
- **Edge Enhancement Mode:** 79% accuracy (79/100 pairs)
- **Difference:** 16% absolute accuracy loss with edge enhancement
- **Performance Drop:** 16.8% relative degradation

### **Error Comparison:**
- **Standard Mode:** Mean error 11.08px, std 51.59px
- **Edge Enhancement Mode:** Mean error 39.71px, std 87.37px
- **Error Increase:** 3.6x higher mean error with edge enhancement
- **Consistency:** Standard mode is more consistent (lower std error)

### **Speed Comparison:**
- **Standard Mode:** 150.76ms mean (slower)
- **Edge Enhancement Mode:** 124.06ms mean (faster)
- **Speedup:** 1.22x faster with edge enhancement
- **Trade-off:** 16% accuracy loss for 18% speed gain

---

## 🔍 **MODE-SPECIFIC FAILURE ANALYSIS**

### **Standard Mode Failures (5 pairs):**
| Pair | Ground Truth | Predicted | Error | Degradation |
|------|-------------|-----------|-------|--------------|
| **pair_014** | (310, 500) | (310, 785) | 285.0px | gaussian_blur, field_nonuniformity |
| **pair_048** | (500, 310) | (668, 454) | 221.3px | gaussian_blur, field_nonuniformity |
| **pair_061** | (650, 350) | (576, 348) | 74.0px | gaussian_blur, field_nonuniformity |
| **pair_064** | (350, 630) | (490, 910) | 313.0px | gaussian_blur, field_nonuniformity |
| **pair_073** | (500, 690) | (689, 785) | 211.5px | gaussian_blur, field_nonuniformity |

### **Edge Enhancement Mode Failures (21 pairs):**
| Pair | Ground Truth | Predicted | Error | Status vs Standard |
|------|-------------|-----------|-------|-------------------|
| **pair_003** | (490, 490) | (420, 490) | 70.0px | New failure |
| **pair_004** | (350, 630) | (420, 630) | 70.0px | New failure |
| **pair_014** | (310, 500) | (309, 785) | 285.0px | Same failure |
| **pair_016** | (310, 500) | (499, 499) | 189.0px | New failure |
| **pair_031** | (500, 499) | (374, 472) | 128.9px | New failure |
| **pair_033** | (500, 500) | (500, 472) | 28.0px | New failure |
| **pair_036** | (499, 310) | (343, 285) | 158.0px | New failure |
| **pair_039** | (500, 500) | (338, 374) | 205.2px | New failure |
| **pair_046** | (500, 310) | (536, 454) | 148.4px | New failure |
| **pair_050** | (500, 310) | (386, 482) | 206.3px | New failure |
| **pair_054** | (500, 310) | (548, 502) | 197.9px | New failure |
| **pair_056** | (504, 310) | (496, 674) | 364.1px | New failure (worst) |
| **pair_058** | (500, 310) | (500, 470) | 160.0px | New failure |
| **pair_061** | (650, 350) | (576, 347) | 74.1px | Same failure |
| **pair_064** | (350, 630) | (350, 350) | 280.0px | Same failure (different error) |
| **pair_067** | (490, 490) | (488, 579) | 89.0px | New failure |
| **pair_072** | (500, 335) | (499, 666) | 331.0px | New failure |
| **pair_073** | (500, 690) | (499, 406) | 284.0px | Same failure (different error) |
| **pair_092** | (500, 310) | (710, 310) | 210.0px | New failure |
| **pair_095** | (500, 500) | (526, 318) | 183.8px | New failure |
| **pair_096** | (500, 310) | (682, 544) | 296.4px | New failure |

**Key Observation:** Edge enhancement mode created 16 new failures while maintaining only 3 of the original 5 failures.

---

## 📈 **PERFORMANCE BREAKDOWN**

### **Error Distribution:**

**Standard Mode:**
- **Perfect (0.0px):** 93 pairs (93%)
- **Near-perfect (1.0-2.2px):** 2 pairs (2%)
- **Failures (>5px):** 5 pairs (5%)
- **Range:** 0.0px - 313.05px

**Edge Enhancement Mode:**
- **Perfect (0.0px):** 72 pairs (72%)
- **Near-perfect (1.0-2.0px):** 7 pairs (7%)
- **Failures (>5px):** 21 pairs (21%)
- **Range:** 0.0px - 364.09px

### **Time Performance:**

**Standard Mode:**
- **Mean:** 150.76ms
- **Median:** 148.17ms
- **Range:** 85.8ms - 265.9ms

**Edge Enhancement Mode:**
- **Mean:** 124.06ms (17.7% faster)
- **Median:** 121.11ms (18.3% faster)
- **Range:** 95.8ms - 159.9ms

---

## 🔬 **TECHNICAL ANALYSIS**

### **Why Edge Enhancement Performed Worse:**

**Edge Enhancement Issues:**
1. ❌ **Noise amplification:** Sobel edge detection amplifies SEM shot noise
2. ❌ **False edges:** Creates spurious edge responses in noisy regions
3. ❌ **Edge/intensity blend:** 60% edge + 40% intensity may not be optimal for this dataset
4. ❌ **Periodic confusion:** Edge enhancement can confuse periodic patterns
5. ❌ **Degradation sensitivity:** More sensitive to gaussian blur + field nonuniformity

**Standard Mode Advantages:**
1. ✅ **Intensity-based:** Relies on original intensity which is more robust
2. ✅ **Noise handling:** Gaussian denoise works better with intensity-only
3. ✅ **NCC optimization:** Designed for intensity-based template matching
4. ✅ **Proven performance:** 95% accuracy validates the approach

### **Failure Pattern Analysis:**

**Common Failure Characteristics in Edge Mode:**
- **New failures:** 16/21 new failures (76%)
- **Large errors:** Many failures >200px (severe periodic confusion)
- **Coordinate errors:** Both X and Y coordinates affected (not just one axis)
- **Degradation sensitivity:** More failures on pairs with complex degradations

**Preserved Successes:**
- **Same patterns:** Edge mode still succeeded on 72 pairs
- **Simple patterns:** Cases with less degradation performed well
- **Consistent locations:** When edge mode succeeded, it was often accurate

---

## 🎯 **PRACTICAL IMPLICATIONS**

### **For Competition Use:**
**Standard mode is clearly superior:**
- ✅ **16% higher accuracy:** 95% vs 79%
- ✅ **3.6x lower mean error:** 11.08px vs 39.71px
- ✅ **More consistent:** Lower std error (51.59px vs 87.37px)
- ❌ **Edge mode trade-off:** 18% speed gain not worth 16% accuracy loss

### **Speed vs. Accuracy Trade-off:**
- **Edge mode:** 18% faster (124ms vs 151ms)
- **Standard mode:** 16% more accurate (95% vs 79%)
- **Trade-off:** Accuracy is more important than speed for competition

### **Recommendation:**
**Use standard mode for competition.** The edge enhancement mode's speed advantage (18%) does not justify the significant accuracy loss (16%), especially when standard mode already achieves excellent performance (95%).

---

## 🏁 **FINAL VERDICT**

### **For Dataset V2 Two Noise 100:**
**Standard mode is the clear winner:**
- ✅ **16% higher accuracy:** 95% vs 79%
- ✅ **3.6x lower mean error:** 11.08px vs 39.71px
- ✅ **More consistent:** Lower std error
- ✅ **Better handling of degradations:** Fewer failures on complex cases

### **Speed Consideration:**
- ⚡ **Edge mode:** 18% faster (124ms vs 151ms)
- ✅ **Standard mode:** Still very fast (151ms = 6.6 FPS)
- ❌ **Edge mode:** Accuracy loss unacceptable for competition

### **Overall Assessment:**
**Standard mode remains the optimal choice for competition use.** The edge enhancement mode's slight speed advantage is not worth the significant accuracy degradation, especially since standard mode already achieves excellent performance (95% accuracy) with acceptable speed.

---

## 📋 **CONCLUSION**

**Dataset V2 Two Noise 100 comparison shows:**
- ✅ **Standard mode:** 95% accuracy, 11.08px mean error (excellent)
- ❌ **Edge enhancement mode:** 79% accuracy, 39.71px mean error (poor)
- ⚡ **Speed trade-off:** 18% faster but 16% accuracy loss
- ✅ **Recommendation:** Use standard mode for competition

**Final assessment:** The edge enhancement mode degrades performance on this dataset, confirming that Sobel edge enhancement is not suitable for DRAM-style patterns with complex degradations. Standard mode remains the optimal choice for the I4C hackathon competition.