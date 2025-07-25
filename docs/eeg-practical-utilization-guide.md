# Practical EEG Utilization Guide: From Theory to Implementation

## Core Concept: Multi-Layer Token Extraction

### What We Can Actually Measure with Neurable EEG

#### 1. Direct Measurements (Raw Data)
- **Voltage fluctuations**: ±100 μV typical range
- **Sampling**: 500Hz (2ms resolution)
- **Channels**: 12 (6 per ear)
- **Frequency range**: 0-131Hz (DC coupled)

#### 2. Derived Measurements (Computed Features)

##### Level 1: Time Domain
```python
# Direct from voltage samples
- Amplitude
- Mean/variance over windows
- Zero crossings
- Peak detection
- Inflection points (d²V/dt² = 0)
```

##### Level 2: Frequency Domain
```python
# Via FFT/Wavelet transforms
- Band powers (delta, theta, alpha, beta, gamma)
- Peak frequencies
- Spectral entropy
- Cross-frequency coupling
```

##### Level 3: Spatial Domain
```python
# Across 12 channels
- Coherence matrices
- Phase locking values
- Spatial patterns/topographies
- Source localization estimates
```

##### Level 4: Nonlinear Dynamics
```python
# From time series analysis
- Lyapunov exponents
- Fractal dimensions
- Entropy measures
- Attractor reconstruction
```

##### Level 5: Advanced Geometric
```python
# From mathematical transformations
- Topological features (via TDA)
- Riemannian trajectories
- Microstate sequences
- Dynamic modes (via DMD)
```

## Practical Token Generation Pipeline

### Stage 1: Microstate Detection (80-120ms)
```python
def detect_current_microstate(eeg_buffer):
    """
    Real-time microstate detection
    """
    # Get current topography
    current_topo = eeg_buffer[:, -1]  # Latest sample across channels
    
    # Compare to canonical microstates
    canonical_states = {
        'A': np.array([...]),  # Right-frontal to left-posterior
        'B': np.array([...]),  # Left-frontal to right-posterior  
        'C': np.array([...]),  # Symmetric frontal
        'D': np.array([...])   # Symmetric central
    }
    
    # Find best match
    correlations = {}
    for state, template in canonical_states.items():
        correlations[state] = np.corrcoef(current_topo, template)[0,1]
    
    current_state = max(correlations, key=correlations.get)
    confidence = correlations[current_state]
    
    return current_state, confidence
```

### Stage 2: Inflection Analysis (Every 10ms)
```python
def track_inflections(eeg_stream):
    """
    Continuous inflection point detection
    """
    inflections = []
    
    for channel in range(12):
        # Second derivative
        d2v = np.gradient(np.gradient(eeg_stream[channel]))
        
        # Zero crossings
        crossings = np.where(np.diff(np.sign(d2v)))[0]
        
        for cross in crossings:
            inflections.append({
                'channel': channel,
                'time': cross / 500.0,
                'type': 'concave_up' if d2v[cross+1] > 0 else 'concave_down',
                'magnitude': abs(eeg_stream[channel, cross])
            })
    
    return inflections
```

### Stage 3: Topological Feature Extraction (300ms windows)
```python
def extract_topological_features(eeg_window):
    """
    TDA for robust pattern detection
    """
    # Create point cloud via delay embedding
    embedded = delay_embed(eeg_window, dim=5, tau=10)
    
    # Compute persistence
    persistence = compute_persistence(embedded)
    
    # Key features
    features = {
        'n_components': len(persistence['H0']),  # Separate regions
        'n_loops': len(persistence['H1']),       # Recurrent patterns
        'max_persistence': max([p['death'] - p['birth'] for p in persistence['H1']]),
        'total_persistence': sum([p['death'] - p['birth'] for p in persistence['H1']])
    }
    
    return features
```

### Stage 4: Consciousness Token Assembly (300ms)
```python
def assemble_consciousness_token(eeg_buffer, start_time):
    """
    Combine all measurements into unified token
    """
    # Get 300ms window
    window = eeg_buffer[:, -150:]  # 150 samples at 500Hz
    
    token = {
        'timestamp': start_time,
        
        # Microstate sequence in this window
        'microstates': extract_microstate_sequence(window),
        
        # Inflection vectors
        'inflections': compute_inflection_vectors(window),
        
        # Topological features
        'topology': extract_topological_features(window),
        
        # Classical features
        'bands': compute_band_powers(window),
        'coherence': compute_coherence_matrix(window),
        'entropy': compute_entropy_measures(window),
        
        # Dynamics
        'lyapunov': estimate_lyapunov(window),
        'attractor_type': classify_attractor(window)
    }
    
    return token
```

## Real-Time Processing Architecture

### Buffer Management
```python
class EEGProcessor:
    def __init__(self, neurable):
        self.neurable = neurable
        self.buffer = CircularBuffer(size=1000)  # 2 seconds
        self.microstate_tracker = MicrostateTracker()
        self.inflection_detector = InflectionDetector()
        self.token_generator = TokenGenerator()
        
    def process_sample(self, sample):
        """
        Called at 500Hz
        """
        # Add to buffer
        self.buffer.add(sample)
        
        # Fast processing (every sample)
        inflection = self.inflection_detector.check(sample)
        if inflection:
            self.handle_inflection(inflection)
        
        # Medium processing (every 50ms)
        if self.buffer.count % 25 == 0:
            microstate = self.microstate_tracker.update(self.buffer.last_n(25))
            
        # Slow processing (every 300ms)  
        if self.buffer.count % 150 == 0:
            token = self.token_generator.create(self.buffer.last_n(150))
            self.handle_token(token)
```

## What Each Method Tells Us

### Microstates
- **What**: Basic building blocks of thought
- **Timescale**: 80-120ms
- **Information**: Current cognitive mode (visual, auditory, memory, executive)
- **Use**: Track rapid thought transitions

### Inflection Points
- **What**: Moments of neural state change
- **Timescale**: Variable (10-100ms)
- **Information**: When brain is changing direction
- **Use**: Precise timing for interventions

### Topological Features
- **What**: Invariant patterns in dynamics
- **Timescale**: 300-1000ms
- **Information**: Underlying structure of consciousness
- **Use**: Robust state classification

### Phase-Amplitude Coupling
- **What**: Cross-frequency interactions
- **Timescale**: 100-500ms
- **Information**: How different rhythms coordinate
- **Use**: Detect binding and integration

### Riemannian Trajectories
- **What**: Path through brain state space
- **Timescale**: 1-5 seconds
- **Information**: How states evolve
- **Use**: Predict future states

## Practical Implementation Schedule

### Week 1: Foundation
- Basic inflection detection ✓
- Simple microstate classification
- Band power extraction
- Real-time visualization

### Week 2: Dynamics
- Phase space reconstruction
- Lyapunov estimation
- Attractor classification
- Chaos measures

### Week 3: Advanced Geometry
- TDA implementation
- Microstate syntax analysis
- Riemannian covariance tracking
- Pattern learning

### Week 4: Integration
- Multi-scale token generation
- Audio glitch timing optimization
- Closed-loop testing
- Personal calibration

## Key Insights for Implementation

1. **All methods work with standard EEG** - no special hardware needed
2. **Different timescales reveal different aspects** - use hierarchical processing
3. **Combine simple and complex** - basic features often most reliable
4. **Personal calibration essential** - brains are unique
5. **Start with known patterns** - eyes open/closed, relaxed/focused

## Memory and Learning

### What to Store
```python
personal_profile = {
    'microstate_templates': {},      # Personal variations
    'inflection_patterns': {},       # Common transitions
    'topological_signatures': {},    # Stable features
    'effective_interventions': {},   # What works
    'baseline_measures': {}          # Normal ranges
}
```

### How to Learn
1. Collect baseline in various states
2. Identify personal patterns
3. Test interventions systematically
4. Track effectiveness over time
5. Adapt parameters accordingly

## The Power of Integration

By combining all these methods, we get:
- **Robustness**: Multiple views of same phenomenon
- **Precision**: Different timescales for different needs
- **Adaptability**: Learn what works for individual
- **Control**: Know when and how to intervene

The Neurable headset, despite only 12 channels, can capture all these features because consciousness signatures are global patterns that manifest even in limited spatial sampling.