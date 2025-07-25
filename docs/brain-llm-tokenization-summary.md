# Brain-to-LLM Tokenization: Research Summary & Validation

## Key Research Findings

### 1. IIT 3.0 Corrections
- **Critical Update**: IIT uses Earth Mover's Distance (EMD), not simple mutual information
- Correct formula: `Φ = min EMD(C(S), ∏C(Mᵢ))`
- PyPhi implementation available for reference
- Warning: Simple systems can have high Φ (doesn't guarantee consciousness)

### 2. Friston's Free Energy Validation
- Variational Free Energy: `F = ⟨L(θ)⟩_q - ⟨ln q(θ)⟩_q`
- Prediction Error is precision-weighted: `PE = ||μ_post - μ_prior||² · Π`
- Dopamine may encode precision (confidence in predictions)
- Active inference adds action through proprioceptive prediction errors

### 3. EEG Preprocessing Best Practices
- **Surprising Finding**: "EEG is better left alone"
- Minimal preprocessing (0.5 Hz high-pass) often outperforms complex pipelines
- ICA can over-clean, removing actual neural signals
- New RELAX pipeline combines MWF + wICA for better artifact removal

## Missing Components We Identified

### 1. Dimensionality Reduction
- Need Vector Quantization (VQ-VAE) for continuous → discrete tokens
- Codebook size: 8192 tokens (matching GPT vocabulary subset)
- No established brain-to-token conversion standard exists

### 2. Temporal Alignment
- EEG: 256-1000 Hz
- WHOOP: 1 Hz  
- P300 events: ~3 Hz
- Solution: Cross-correlation with 5-10s autonomic delay

### 3. Validation Metrics
- No standard benchmarks for consciousness tokenization
- Existing BCI metrics focus on classification accuracy, not semantic preservation
- Need new metrics for thought-to-token fidelity

## Implementation Recommendations

### 1. Start Simple
- Use minimal EEG preprocessing (0.5 Hz high-pass only)
- Focus on P300 detection first (most reliable consciousness marker)
- Add complexity only if simple approach fails

### 2. Critical Parameters
- P300 window: 250-500ms post-stimulus
- Gamma band: 50-100 Hz for semantic content
- Gabor wavelet parameters: Need empirical optimization
- Token vocabulary: Start with 8192, adjust based on results

### 3. Validation Strategy
- Compare tokens from same thought across sessions
- Test semantic similarity preservation
- Validate against behavioral outcomes
- Use existing BCI tasks as baseline

## Next Steps

1. **Implement P300 detector** with minimal preprocessing
2. **Test holonomic transform** with various Gabor parameters  
3. **Build VQ-VAE** for tokenization with 8192 codebook
4. **Create validation dataset** with known thought-behavior pairs
5. **Compare against existing EEG-to-text methods**

## Key Insights

The math is largely correct, but:
- IIT needs EMD implementation (use PyPhi)
- Free energy needs precision weighting
- Less preprocessing is more
- No standard tokenization exists - we're pioneering

The framework is theoretically sound but needs empirical validation.