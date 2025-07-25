# Robustness & Edge Case Handling for Brain-to-LLM Tokenization

## 1. Missing Data Handling

### WHOOP Data Interruption
```python
class RobustSomaticProcessor:
    def __init__(self, buffer_minutes=30):
        self.buffer = CircularBuffer(buffer_minutes * 60)  # 30 min history
        self.last_valid = None
        self.interpolator = None
        
    def process(self, whoop_data):
        """Handle missing WHOOP data gracefully"""
        if whoop_data is None or not self.is_valid(whoop_data):
            return self.handle_missing()
        
        # Update buffer with valid data
        self.buffer.add(whoop_data)
        self.last_valid = whoop_data
        self.update_interpolator()
        
        return self.compute_somatic_marker(whoop_data)
    
    def handle_missing(self):
        """Strategies for missing data"""
        strategies = {
            'last_value': self.use_last_valid,
            'interpolate': self.interpolate_missing,
            'predict': self.predict_from_eeg,
            'default': self.use_population_baseline
        }
        
        # Try strategies in order
        for strategy_name, strategy_func in strategies.items():
            try:
                result = strategy_func()
                if result is not None:
                    return result
            except:
                continue
        
        # Ultimate fallback
        return 0.5  # Neutral somatic state
    
    def interpolate_missing(self):
        """Interpolate based on recent history"""
        if len(self.buffer) < 10:
            return None
            
        # Fit ARIMA model on recent data
        from statsmodels.tsa.arima.model import ARIMA
        
        history = np.array(self.buffer.get_all())
        model = ARIMA(history, order=(2,1,2))
        model_fit = model.fit(disp=0)
        
        # Predict next value
        forecast = model_fit.forecast(steps=1)[0]
        
        # Bound to valid range
        return np.clip(forecast, 0, 1)
```

### EEG Artifact Detection & Recovery
```python
class ArtifactRobustEEG:
    def __init__(self, n_channels=64, fs=256):
        self.n_channels = n_channels
        self.fs = fs
        self.artifact_detector = self.build_artifact_detector()
        
    def build_artifact_detector(self):
        """Multi-method artifact detection"""
        return {
            'amplitude': function x: np.abs(x) > 100,  # μV
            'gradient': function x: np.abs(np.gradient(x)) > 50,
            'variance': function x: np.var(x) > 1000,
            'kurtosis': function x: scipy.stats.kurtosis(x) > 10,
            'power_ratio': function x: self.high_freq_power(x) > 0.5
        }
    
    def process_with_artifacts(self, eeg_chunk):
        """Handle artifacts channel by channel"""
        clean_channels = []
        artifact_mask = np.zeros(self.n_channels, dtype=bool)
        
        for ch in range(self.n_channels):
            signal = eeg_chunk[ch]
            
            # Check all artifact criteria
            is_artifact = any(
                detector(signal) 
                for detector in self.artifact_detector.values()
            )
            
            if is_artifact:
                artifact_mask[ch] = True
                # Interpolate from neighbors
                clean_signal = self.interpolate_channel(eeg_chunk, ch)
                clean_channels.append(clean_signal)
            else:
                clean_channels.append(signal)
        
        # If too many channels bad, use previous good segment
        if artifact_mask.sum() > self.n_channels * 0.3:
            return self.use_previous_clean()
        
        return np.array(clean_channels), artifact_mask
    
    def interpolate_channel(self, eeg_chunk, bad_channel):
        """Spherical spline interpolation"""
        # Get channel positions (assuming 10-20 system)
        positions = self.get_channel_positions()
        
        # Find nearest neighbors
        distances = np.sqrt(
            (positions[:, 0] - positions[bad_channel, 0])**2 +
            (positions[:, 1] - positions[bad_channel, 1])**2
        )
        
        # Use 4 nearest neighbors
        neighbors = np.argsort(distances)[1:5]
        
        # Distance-weighted average
        weights = 1 / distances[neighbors]
        weights /= weights.sum()
        
        interpolated = np.sum(
            eeg_chunk[neighbors] * weights[:, np.newaxis],
            axis=0
        )
        
        return interpolated
```

## 2. State Transition Handling

### Ambiguous State Detection
```python
class StateTransitionHandler:
    def __init__(self, transition_threshold=0.3):
        self.transition_threshold = transition_threshold
        self.state_history = deque(maxlen=10)
        self.transition_detector = self.build_transition_detector()
        
    def detect_transition(self, current_features):
        """Detect if in transition between states"""
        if len(self.state_history) < 2:
            return False, None
        
        # Compute state probabilities
        state_probs = self.compute_state_probabilities(current_features)
        
        # Check if multiple states have similar probability
        sorted_probs = sorted(state_probs.values(), reverse=True)
        if len(sorted_probs) >= 2:
            prob_diff = sorted_probs[0] - sorted_probs[1]
            
            if prob_diff < self.transition_threshold:
                # Ambiguous state - likely in transition
                return True, self.handle_transition(state_probs)
        
        return False, max(state_probs.items(), key=function x: x[1])[0]
    
    def handle_transition(self, state_probs):
        """Generate transition token"""
        # Sort states by probability
        sorted_states = sorted(
            state_probs.items(), 
            key=function x: x[1], 
            reverse=True
        )
        
        # Create composite token
        from_state = self.state_history[-1]
        to_state = sorted_states[0][0]
        confidence = sorted_states[0][1]
        
        # Special transition token format
        transition_token = self.encode_transition(
            from_state, to_state, confidence
        )
        
        return ('TRANSITION', transition_token)
    
    def smooth_state_sequence(self, states, window=5):
        """Smooth state assignments using HMM"""
        from hmmlearn import hmm
        
        # Build transition probability matrix from history
        n_states = len(self.state_space)
        trans_mat = self.estimate_transition_matrix()
        
        # Hidden Markov Model smoothing
        model = hmm.GaussianHMM(
            n_components=n_states,
            transmat=trans_mat,
            init_params=""
        )
        
        # Viterbi decoding for most likely sequence
        log_prob, smoothed_states = model.decode(
            np.array(states).reshape(-1, 1)
        )
        
        return smoothed_states
```

## 3. Individual Calibration

### Personal Model Adaptation
```python
class PersonalizedTokenizer:
    def __init__(self, user_id):
        self.user_id = user_id
        self.calibration_data = []
        self.personal_model = None
        self.population_model = self.load_population_model()
        
    def calibrate(self, calibration_session):
        """
        Calibration protocol:
        1. Rest state (2 min)
        2. Attention tasks (Stroop, n-back)
        3. Emotional induction (IAPS images)
        4. Motor tasks
        """
        tasks = {
            'rest': self.calibrate_rest,
            'attention': self.calibrate_attention,
            'emotion': self.calibrate_emotion,
            'motor': self.calibrate_motor
        }
        
        calibration_features = {}
        
        for task_name, task_func in tasks.items():
            print(f"Starting {task_name} calibration...")
            features = task_func(calibration_session)
            calibration_features[task_name] = features
            
        # Build personal model
        self.personal_model = self.build_personal_model(calibration_features)
        
        # Compute adaptation parameters
        self.adaptation_params = self.compute_adaptation(
            self.personal_model,
            self.population_model
        )
        
        return self.adaptation_params
    
    def compute_adaptation(self, personal, population):
        """Compute how to adapt population model to individual"""
        adaptations = {}
        
        # Feature scaling factors
        for feature in ['alpha', 'beta', 'gamma', 'p300', 'hrv']:
            personal_mean = personal[feature]['mean']
            personal_std = personal[feature]['std']
            
            pop_mean = population[feature]['mean']
            pop_std = population[feature]['std']
            
            # Z-score normalization parameters
            adaptations[feature] = {
                'scale': pop_std / personal_std,
                'shift': pop_mean - (personal_mean * pop_std / personal_std)
            }
        
        # State thresholds
        for state in ['attention', 'rest', 'arousal']:
            personal_threshold = personal[state]['threshold']
            pop_threshold = population[state]['threshold']
            
            adaptations[f'{state}_threshold'] = {
                'multiplier': pop_threshold / personal_threshold
            }
        
        return adaptations
    
    def adaptive_tokenize(self, features):
        """Tokenize using personalized model"""
        if self.personal_model is None:
            # Fall back to population model
            return self.population_tokenize(features)
        
        # Apply personal adaptations
        adapted_features = {}
        for feat_name, feat_value in features.items():
            if feat_name in self.adaptation_params:
                scale = self.adaptation_params[feat_name]['scale']
                shift = self.adaptation_params[feat_name]['shift']
                adapted_features[feat_name] = feat_value * scale + shift
            else:
                adapted_features[feat_name] = feat_value
        
        # Use adapted features for tokenization
        return self.tokenize(adapted_features)
```

## 4. Real-Time Performance Constraints

### Adaptive Processing Pipeline
```python
class AdaptiveRealTimeProcessor:
    def __init__(self, target_latency_ms=100):
        self.target_latency = target_latency_ms / 1000.0
        self.processing_times = deque(maxlen=100)
        self.quality_levels = {
            'full': self.process_full_quality,
            'reduced': self.process_reduced_quality,
            'minimal': self.process_minimal_quality
        }
        self.current_quality = 'full'
        
    def process_adaptive(self, data):
        """Adapt processing quality to meet latency requirements"""
        start_time = time.perf_counter()
        
        # Try current quality level
        result = self.quality_levels[self.current_quality](data)
        
        # Measure processing time
        proc_time = time.perf_counter() - start_time
        self.processing_times.append(proc_time)
        
        # Adapt quality level
        self.adapt_quality_level()
        
        return result, self.current_quality
    
    def adapt_quality_level(self):
        """Adjust processing quality based on performance"""
        if len(self.processing_times) < 10:
            return
        
        avg_time = np.mean(self.processing_times)
        
        if avg_time > self.target_latency * 1.2:
            # Too slow, reduce quality
            if self.current_quality == 'full':
                self.current_quality = 'reduced'
            elif self.current_quality == 'reduced':
                self.current_quality = 'minimal'
                
        elif avg_time < self.target_latency * 0.5:
            # Fast enough, try increasing quality
            if self.current_quality == 'minimal':
                self.current_quality = 'reduced'
            elif self.current_quality == 'reduced':
                self.current_quality = 'full'
    
    def process_full_quality(self, data):
        """Full quality processing"""
        features = {
            'fft': compute_fft_all_channels(data),
            'connectivity': compute_full_connectivity(data),
            'phi': compute_exact_phi(data),
            'wavelets': compute_wavelet_transform(data)
        }
        return self.full_tokenization(features)
    
    def process_reduced_quality(self, data):
        """Reduced quality - skip expensive computations"""
        features = {
            'fft': compute_fft_subset_channels(data, n=16),
            'connectivity': compute_connectivity_approximation(data),
            'phi': approximate_phi_spectral(data),
            'wavelets': None  # Skip wavelets
        }
        return self.reduced_tokenization(features)
    
    def process_minimal_quality(self, data):
        """Minimal quality - only essential features"""
        features = {
            'power_bands': compute_band_power_fast(data),
            'p300_check': quick_p300_detection(data),
            'variance': np.var(data, axis=1)
        }
        return self.minimal_tokenization(features)
```

## 5. Confidence Scoring & Uncertainty

### Uncertainty Quantification
```python
class UncertaintyAwareTokenizer:
    def __init__(self, n_models=10):
        # Ensemble of models for uncertainty
        self.models = [
            self.create_model(seed=i) 
            for i in range(n_models)
        ]
        self.uncertainty_threshold = 0.3
        
    def tokenize_with_uncertainty(self, features):
        """Generate token with uncertainty estimate"""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            token = model.predict_token(features)
            predictions.append(token)
        
        predictions = np.array(predictions)
        
        # Compute uncertainty metrics
        # 1. Token agreement (discrete uncertainty)
        unique_tokens, counts = np.unique(predictions, return_counts=True)
        mode_token = unique_tokens[np.argmax(counts)]
        token_entropy = scipy.stats.entropy(counts / len(predictions))
        
        # 2. Feature-level uncertainty
        feature_uncertainties = self.compute_feature_uncertainties(features)
        
        # 3. Temporal uncertainty (state stability)
        temporal_uncertainty = self.compute_temporal_uncertainty()
        
        # Combine uncertainties
        total_uncertainty = self.combine_uncertainties({
            'token_entropy': token_entropy,
            'feature': np.mean(feature_uncertainties),
            'temporal': temporal_uncertainty
        })
        
        # Decision based on uncertainty
        if total_uncertainty > self.uncertainty_threshold:
            return self.handle_high_uncertainty(
                mode_token, total_uncertainty, predictions
            )
        
        return {
            'token': mode_token,
            'confidence': 1 - total_uncertainty,
            'uncertainty_components': {
                'aleatoric': token_entropy,  # Data uncertainty
                'epistemic': np.mean(feature_uncertainties),  # Model uncertainty
                'temporal': temporal_uncertainty
            }
        }
    
    def compute_feature_uncertainties(self, features):
        """Compute uncertainty for each feature"""
        uncertainties = {}
        
        for feat_name, feat_value in features.items():
            # Signal quality metrics
            snr = self.compute_snr(feat_value)
            
            # Deviation from expected range
            z_score = self.compute_feature_zscore(feat_name, feat_value)
            
            # Combine into uncertainty
            uncertainties[feat_name] = 1 / (1 + snr) + sigmoid(np.abs(z_score) - 3)
        
        return uncertainties
    
    def handle_high_uncertainty(self, token, uncertainty, predictions):
        """Special handling for uncertain tokens"""
        # Option 1: Use special uncertainty tokens
        uncertainty_level = int(uncertainty * 10)  # 0-10 scale
        uncertain_token = self.create_uncertainty_token(
            token, uncertainty_level
        )
        
        # Option 2: Return distribution instead of single token
        token_distribution = np.bincount(predictions) / len(predictions)
        
        return {
            'token': uncertain_token,
            'confidence': 1 - uncertainty,
            'distribution': token_distribution,
            'raw_token': token,
            'action': 'uncertainty_marked'
        }
```

## 6. System Health Monitoring

### Runtime Diagnostics
```python
class SystemHealthMonitor:
    def __init__(self):
        self.metrics = {
            'latency': deque(maxlen=1000),
            'token_rate': deque(maxlen=1000),
            'artifact_rate': deque(maxlen=1000),
            'uncertainty_rate': deque(maxlen=1000),
            'data_quality': deque(maxlen=1000)
        }
        self.alerts = []
        
    def update_metrics(self, processing_result):
        """Update system health metrics"""
        self.metrics['latency'].append(processing_result['latency'])
        self.metrics['token_rate'].append(processing_result['tokens_per_second'])
        self.metrics['artifact_rate'].append(processing_result['artifact_ratio'])
        self.metrics['uncertainty_rate'].append(processing_result['uncertainty'])
        self.metrics['data_quality'].append(processing_result['quality_score'])
        
        # Check for issues
        self.check_health()
    
    def check_health(self):
        """Check system health and raise alerts"""
        # Latency check
        avg_latency = np.mean(self.metrics['latency'])
        if avg_latency > 150:  # ms
            self.raise_alert('HIGH_LATENCY', f'Average latency {avg_latency}ms')
        
        # Data quality check
        avg_quality = np.mean(self.metrics['data_quality'])
        if avg_quality < 0.7:
            self.raise_alert('LOW_DATA_QUALITY', f'Quality score {avg_quality}')
        
        # Artifact rate check
        artifact_rate = np.mean(self.metrics['artifact_rate'])
        if artifact_rate > 0.3:
            self.raise_alert('HIGH_ARTIFACTS', f'Artifact rate {artifact_rate}')
        
        # Token production rate
        token_rate = np.mean(self.metrics['token_rate'])
        if token_rate < 2.0:  # tokens/second
            self.raise_alert('LOW_TOKEN_RATE', f'Token rate {token_rate}/s')
    
    def get_health_report(self):
        """Generate comprehensive health report"""
        return {
            'status': 'healthy' if len(self.alerts) == 0 else 'degraded',
            'metrics_summary': {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': self.compute_trend(values)
                }
                for metric, values in self.metrics.items()
            },
            'active_alerts': self.alerts,
            'recommendations': self.generate_recommendations()
        }
```

## 7. Graceful Degradation Strategy

### Fallback Hierarchy
```python
FALLBACK_HIERARCHY = [
    {
        'name': 'full_multimodal',
        'requirements': ['eeg', 'whoop', 'low_artifacts'],
        'token_quality': 1.0
    },
    {
        'name': 'eeg_only',
        'requirements': ['eeg', 'low_artifacts'],
        'token_quality': 0.8
    },
    {
        'name': 'minimal_eeg',
        'requirements': ['eeg'],
        'token_quality': 0.6
    },
    {
        'name': 'whoop_prediction',
        'requirements': ['whoop'],
        'token_quality': 0.4
    },
    {
        'name': 'null_tokens',
        'requirements': [],
        'token_quality': 0.0
    }
]

def select_processing_mode(available_data):
    """Select best processing mode given available data"""
    for mode in FALLBACK_HIERARCHY:
        if all(req in available_data for req in mode['requirements']):
            return mode
    
    return FALLBACK_HIERARCHY[-1]  # Worst case
```

This comprehensive robustness framework ensures the brain-to-LLM tokenization system can handle real-world conditions including missing data, artifacts, individual differences, and computational constraints while maintaining transparency about confidence and system health.