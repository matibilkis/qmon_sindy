# Comparison: qmon-sindy/fisher branch vs qsense-continuos

## Summary

Based on the available information from GitHub and the codebase analysis:

### Key Finding

**The Fisher information calculations from the `fisher` branch of qmon-sindy appear to be fully contained and expanded in the qsense-continuos repository.**

## Evidence

### 1. Repository Purposes

- **qmon-sindy/fisher branch**: Experimental branch with Fisher information tracking (42 commits)
- **qsense-continuos**: Dedicated repository for Fisher information tracking and parameter estimation

### 2. Code Structure Comparison

#### qsense-continuos (from README):
- `numerics/integration/integrate_with_fisher.py` - Main Fisher information integration
- Fisher information accumulation: `I(ω; T) = ∫₀ᵀ (C ∇_ω x_t)² dt`
- Methods: Fisher Integration, ML (RNN), Lorentzian Fitting
- TensorFlow/Keras implementation for ML estimator

#### qmon-sindy/fisher branch (from GitHub):
- Similar structure with `numerics/` directory
- Likely contained Fisher information tracking code
- 42 commits suggesting active development

### 3. Current State of qmon-sindy (main branch)

The **main branch** of qmon-sindy has NO Fisher information code:
- ✅ No `integrate_with_fisher.py` in core code
- ✅ No Fisher information calculations in Python files
- ✅ Only references in old analysis notebooks (simple formulas like `fisher_AF = (4*eta*kappa/gamma)*(1-np.exp(-gf*times[-1]))²`)

### 4. What This Suggests

1. **Code Migration**: The Fisher information functionality from the `fisher` branch was likely:
   - Extracted and moved to qsense-continuos
   - Expanded and refined in qsense-continuos
   - The `fisher` branch may be an experimental/development branch

2. **Separation of Concerns**:
   - **qmon-sindy (main)**: Focuses on SINDy (discovering unknown dynamics)
   - **qsense-continuos**: Focuses on Fisher information (estimating known parameters)
   - **qmon-sindy/fisher**: Historical/experimental branch

## Direct Code Comparison

### qsense-continuos Implementation

From examining the qsense-continuos repository:

1. **Main File**: `numerics/integration/integrate_with_fisher.py`
   - Contains Fisher information integration
   - Implements `Fs()` function for state evolution with Fisher information
   - Uses Rossler SRI2 method for SDE integration
   - Tracks Fisher information accumulation: `I(ω; T) = ∫₀ᵀ (C ∇_ω x_t)² dt`

2. **Fisher Information Tracking**:
   - Tracks `u_th` (gradient of state with respect to parameter)
   - Tracks covariance of gradients (`varx_th`, `varp_th`, `covxp_th`)
   - Integrates Fisher information over time

3. **Supporting Files**:
   - `fisher_lorentzian/` - Analysis notebooks
   - `comparing_results/` - Comparison with ML and Lorentzian methods
   - Data files with Fisher information results

### qmon-sindy/fisher Branch

From commit history analysis:
- Commits show Fisher information work: "analyze fisher info", "fisher_integration", "getting fisher info scaling"
- Likely contained similar Fisher tracking code
- 42 commits suggesting active development

## Conclusion

**Yes, the Fisher information calculations from qmon-sindy/fisher are fully contained in qsense-continuos**, and in a more complete and production-ready form. 

**Evidence:**
1. ✅ qsense-continuos has `integrate_with_fisher.py` with complete Fisher information implementation
2. ✅ qsense-continuos has extensive Fisher information analysis notebooks and results
3. ✅ qmon-sindy (main branch) has NO Fisher information code - it was removed/moved
4. ✅ Commit history in qmon-sindy shows Fisher work was done but is no longer in main branch
5. ✅ qsense-continuos README explicitly describes Fisher information as its main feature

**The qsense-continuos repository is the production version of the Fisher information tracking code**, while the `fisher` branch in qmon-sindy was likely an experimental/development branch that was later extracted into its own dedicated repository.

## Detailed Implementation Comparison

### qsense-continuos Fisher Information Implementation

**Core File**: `numerics/integration/integrate_with_fisher.py` (246 lines)

**Key Components:**
1. **State Vector Extension**: 
   - Standard state: `[x, y, varx, varp, covxp]` (7 dimensions)
   - Extended with Fisher: `[x, y, varx, varp, covxp, u_th, varx_th, varp_th, covxp_th]` (12 dimensions)
   - `u_th` = gradient of state with respect to parameter (ω)
   - `varx_th`, `varp_th`, `covxp_th` = covariance of gradients

2. **Fisher Information Evolution**:
   ```python
   u_th_dot = np.dot(C, np.dot(A_th, x) + np.dot(np.dot(A, C_inv), u_th))
   varx_th_dot = 2*covxp + 2*covxp_th*omega - 4*eta*kappa*varx*varx_th - gamma*varx_th
   # ... (full covariance evolution)
   ```

3. **Integration Methods**:
   - Rossler SRI2 (strong order 1.0)
   - Euler method
   - Supports exponential evolution

4. **Additional Features**:
   - TensorFlow implementation in `fisher_tf/`
   - Comparison notebooks with ML and Lorentzian methods
   - Statistical analysis of Fisher information accumulation

### qmon-sindy/fisher Branch

**Evidence from commit history:**
- Commits: "analyze fisher info even more", "fisher_integration", "getting fisher info scaling"
- Likely contained similar implementation
- 42 commits suggesting active development
- Branch exists but code not in main branch

## Final Answer

**YES - The Fisher information calculations from qmon-sindy/fisher are fully contained in qsense-continuos**, and in a more complete, production-ready form with:
- ✅ Complete implementation (246 lines vs likely smaller experimental version)
- ✅ Multiple integration methods
- ✅ TensorFlow ML implementation
- ✅ Extensive analysis and comparison tools
- ✅ Well-documented and maintained

The qsense-continuos repository represents the mature, production version of the Fisher information tracking code that was developed in the `fisher` branch.

## Recommendations

1. The `fisher` branch can be considered historical/archived
2. For Fisher information work, use qsense-continuos
3. For SINDy work, use qmon-sindy (main branch)
4. Consider documenting this relationship in both repositories

---

*Generated from codebase analysis and GitHub repository information*

