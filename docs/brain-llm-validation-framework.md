# Brain-to-LLM Tokenization Validation Framework

## Overview
This framework tests the consistency and reliability of brain-to-token conversion using the most consistent features identified in our research.

## 1. P300 Detector Implementation with Minimal Preprocessing

### Minimal Preprocessing Pipeline
```python
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import mne

class MinimalP300Detector:
    def __init__(self, fs=256, highpass=0.5):
        """
        P300 detector with minimal preprocessing
        Based on research showing "EEG is better left alone"
        """
        self.fs = fs
        self.highpass = highpass
        
        # Design minimal high-pass filter (0.5 Hz)
        self.b_hp, self.a_hp = signal.butter(
            2, highpass / (fs/2), 'high'
        )
        
        # P300 parameters
        self.p300_window = (0.250, 0.500)  # 250-500ms
        self.baseline_window = (-0.200, 0)  # -200 to 0ms
        
    def preprocess_minimal(self, eeg_data):
        """
        Minimal preprocessing - just 0.5 Hz high-pass
        """
        # Apply high-pass filter
        filtered = signal.filtfilt(
            self.b_hp, self.a_hp, eeg_data, axis=1
        )
        
        # That's it! No ICA, no complex artifact removal
        return filtered
    
    def detect_p300_epochs(self, eeg_data, event_markers, channel_names=None):
        """
        Detect P300 from event-locked epochs
        """
        # Minimal preprocessing
        clean_data = self.preprocess_minimal(eeg_data)
        
        # Extract epochs around events
        epochs = []
        baseline_start = int(self.baseline_window[0] * self.fs)
        baseline_end = int(self.baseline_window[1] * self.fs)
        p300_start = int(self.p300_window[0] * self.fs)
        p300_end = int(self.p300_window[1] * self.fs)
        
        for event_idx in event_markers:
            # Check bounds
            if (event_idx + baseline_start >= 0 and 
                event_idx + p300_end < clean_data.shape[1]):
                
                # Extract epoch
                epoch = clean_data[:, event_idx + baseline_start:
                                     event_idx + p300_end]
                
                # Baseline correction
                baseline = epoch[:, :baseline_end-baseline_start].mean(axis=1)
                epoch_corrected = epoch - baseline[:, np.newaxis]
                
                epochs.append(epoch_corrected)
        
        epochs = np.array(epochs)
        
        # Average across trials
        erp = epochs.mean(axis=0)
        
        # Find P300 (positive peak in posterior channels)
        p300_results = self.find_p300_component(erp, channel_names)
        
        return p300_results, erp, epochs
    
    def find_p300_component(self, erp, channel_names=None):
        """
        Identify P300 component in ERP
        """
        # Select posterior channels (Pz, P3, P4, POz)
        if channel_names:
            posterior_idx = [
                i for i, ch in enumerate(channel_names)
                if any(p in ch.upper() for p in ['PZ', 'P3', 'P4', 'POZ'])
            ]
        else:
            # Assume 64-channel layout, take central posterior
            posterior_idx = [30, 31, 32, 47, 48, 49]  # Approximate
        
        # P300 time window in samples
        p300_start_idx = int((self.p300_window[0] + 0.2) * self.fs)
        p300_end_idx = int((self.p300_window[1] + 0.2) * self.fs)
        
        # Average posterior channels
        posterior_signal = erp[posterior_idx].mean(axis=0)
        
        # Find peaks in P300 window
        p300_segment = posterior_signal[p300_start_idx:p300_end_idx]
        
        # Adaptive threshold: 2 standard deviations above baseline
        baseline_std = posterior_signal[:int(0.2 * self.fs)].std()
        threshold = 2 * baseline_std
        
        # Find peaks
        peaks, properties = find_peaks(
            p300_segment,
            height=threshold,
            distance=int(0.050 * self.fs)  # Min 50ms between peaks
        )
        
        if len(peaks) > 0:
            # Get largest peak
            max_idx = np.argmax(properties['peak_heights'])
            p300_amplitude = properties['peak_heights'][max_idx]
            p300_latency = (peaks[max_idx] + p300_start_idx) / self.fs - 0.2
            
            return {
                'detected': True,
                'amplitude': p300_amplitude,
                'amplitude_normalized': p300_amplitude / baseline_std,
                'latency': p300_latency,
                'confidence': min(p300_amplitude / (3 * baseline_std), 1.0)
            }
        else:
            return {
                'detected': False,
                'amplitude': 0,
                'amplitude_normalized': 0,
                'latency': None,
                'confidence': 0
            }
    
    def validate_consistency(self, eeg_sessions, event_markers_list):
        """
        Validate P300 consistency across sessions
        """
        session_results = []
        
        for session_idx, (eeg_data, events) in enumerate(
            zip(eeg_sessions, event_markers_list)
        ):
            results, erp, epochs = self.detect_p300_epochs(eeg_data, events)
            
            session_results.append({
                'session': session_idx,
                'p300': results,
                'erp': erp,
                'n_trials': len(epochs),
                'snr': self.compute_snr(erp, results)
            })
        
        # Compute inter-session consistency
        consistency = self.compute_consistency_metrics(session_results)
        
        return session_results, consistency
    
    def compute_snr(self, erp, p300_results):
        """
        Compute signal-to-noise ratio
        """
        if not p300_results['detected']:
            return 0
        
        # Signal: P300 amplitude
        signal_power = p300_results['amplitude'] ** 2
        
        # Noise: Baseline variance
        baseline_end = int(0.2 * self.fs)
        noise_power = np.var(erp[:, :baseline_end])
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def compute_consistency_metrics(self, session_results):
        """
        Compute consistency metrics across sessions
        """
        # Extract features
        amplitudes = [s['p300']['amplitude_normalized'] 
                     for s in session_results if s['p300']['detected']]
        latencies = [s['p300']['latency'] 
                    for s in session_results if s['p300']['detected']]
        
        if len(amplitudes) < 2:
            return {'error': 'Not enough P300 detections for consistency'}
        
        # Consistency metrics
        consistency = {
            'amplitude_consistency': 1 - np.std(amplitudes) / np.mean(amplitudes),
            'latency_consistency': 1 - np.std(latencies) / np.mean(latencies),
            'detection_rate': len(amplitudes) / len(session_results),
            'mean_snr': np.mean([s['snr'] for s in session_results]),
            'amplitude_correlation': self.compute_correlation(amplitudes),
            'latency_stability_ms': np.std(latencies) * 1000
        }
        
        # Overall consistency score (weighted average)
        consistency['overall_score'] = (
            0.4 * consistency['amplitude_consistency'] +
            0.3 * consistency['detection_rate'] +
            0.2 * consistency['latency_consistency'] +
            0.1 * (consistency['mean_snr'] / 20)  # Normalize SNR
        )
        
        return consistency
    
    def compute_correlation(self, values):
        """
        Compute average pairwise correlation
        """
        if len(values) < 2:
            return 0
        
        correlations = []
        for i in range(len(values)-1):
            for j in range(i+1, len(values)):
                correlations.append(np.corrcoef([values[i]], [values[j]])[0, 1])
        
        return np.mean(correlations)
```

## 2. Holonomic Transform Validation

### Gabor Parameter Optimization
```python
class HolonomicTransformValidator:
    def __init__(self, electrode_positions=None):
        """
        Validate holonomic transform with different Gabor parameters
        """
        if electrode_positions is None:
            # Default 10-20 system for 64 channels
            self.positions = self.generate_10_20_positions()
        else:
            self.positions = electrode_positions
            
        # Parameter ranges to test
        self.freq_range = np.logspace(-1, 1.7, 10)  # 0.1 to 50 Hz
        self.sigma_range = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        self.n_orientations = [4, 6, 8, 12]
        
    def generate_10_20_positions(self):
        """
        Generate approximate 10-20 electrode positions
        """
        # Simplified 8x8 grid
        x = np.linspace(-1, 1, 8)
        y = np.linspace(-1, 1, 8)
        xx, yy = np.meshgrid(x, y)
        
        # Mask to approximate head shape
        mask = xx**2 + yy**2 <= 1
        positions = np.column_stack([xx[mask], yy[mask]])
        
        return positions
    
    def optimize_gabor_parameters(self, eeg_data, behavioral_labels):
        """
        Find optimal Gabor parameters for behavioral prediction
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        best_score = 0
        best_params = None
        
        # Grid search over parameters
        for n_orient in self.n_orientations:
            for sigma in self.sigma_range:
                # Extract features with current parameters
                features = self.extract_gabor_features(
                    eeg_data, 
                    n_orientations=n_orient,
                    sigma=sigma
                )
                
                # Test behavioral prediction
                scores = cross_val_score(
                    RandomForestClassifier(n_estimators=100),
                    features,
                    behavioral_labels,
                    cv=5,
                    scoring='accuracy'
                )
                
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'n_orientations': n_orient,
                        'sigma': sigma,
                        'score': mean_score,
                        'std': scores.std()
                    }
                
                print(f"n_orient={n_orient}, sigma={sigma}: "
                      f"score={mean_score:.3f} ± {scores.std():.3f}")
        
        return best_params
    
    def extract_gabor_features(self, eeg_data, n_orientations=8, 
                               sigma=2.0, n_scales=5):
        """
        Extract Gabor wavelet features
        """
        n_samples, n_channels, n_time = eeg_data.shape
        
        # Frequency scales (log-spaced)
        frequencies = np.logspace(-1, 1.5, n_scales)
        orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)
        
        # Feature array
        features = np.zeros((n_samples, n_channels * n_orientations * n_scales))
        
        for sample_idx in range(n_samples):
            # Interpolate to 2D grid
            grid_data = self.interpolate_to_grid(
                eeg_data[sample_idx], 
                self.positions
            )
            
            feature_idx = 0
            for freq in frequencies:
                for orient in orientations:
                    # Create Gabor kernel
                    kernel = self.create_gabor_kernel(
                        grid_data.shape,
                        freq,
                        orient,
                        sigma
                    )
                    
                    # Convolve
                    response = np.abs(signal.convolve2d(
                        grid_data,
                        kernel,
                        mode='same'
                    ))
                    
                    # Store max response per channel
                    features[sample_idx, feature_idx:feature_idx+n_channels] = \
                        response.max(axis=0)[:n_channels]
                    
                    feature_idx += n_channels
        
        return features
    
    def create_gabor_kernel(self, shape, frequency, orientation, sigma):
        """
        Create 2D Gabor kernel
        """
        rows, cols = shape
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
        Y_rot = -X * np.sin(orientation) + Y * np.cos(orientation)
        
        # Gabor function
        gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * frequency * X_rot)
        
        gabor = gaussian * sinusoid
        
        # Normalize
        gabor -= gabor.mean()
        gabor /= np.sqrt(np.sum(gabor**2))
        
        return gabor
    
    def interpolate_to_grid(self, channel_data, positions, grid_size=32):
        """
        Interpolate irregular electrode positions to regular grid
        """
        from scipy.interpolate import griddata
        
        # Create regular grid
        xi = np.linspace(-1, 1, grid_size)
        yi = np.linspace(-1, 1, grid_size)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate each time point
        n_time = channel_data.shape[1]
        grid_data = np.zeros((grid_size, grid_size, n_time))
        
        for t in range(n_time):
            grid_data[:, :, t] = griddata(
                positions,
                channel_data[:, t],
                (xi, yi),
                method='cubic',
                fill_value=0
            )
        
        return grid_data
```

## 3. VQ-VAE Implementation for Brain States

### Vector Quantized Autoencoder for Tokenization
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings=8192, embedding_dim=256, beta=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        # Initialize codebook
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/n_embeddings, 1/n_embeddings)
        
    def forward(self, z):
        """
        z: (batch, channels, height, width) or (batch, features)
        """
        # Flatten input
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Compute distances to codebook entries
        distances = (z_flattened.pow(2).sum(1, keepdim=True) 
                    - 2 * z_flattened @ self.embedding.weight.t()
                    + self.embedding.weight.pow(2).sum(1, keepdim=True).t())
        
        # Get nearest codebook entries
        encoding_indices = distances.argmin(1).unsqueeze(1)
        
        # Quantize
        quantized = self.embedding(encoding_indices).view_as(z)
        
        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        # Perplexity
        avg_probs = (encoding_indices.float().mean(0))
        perplexity = (-(avg_probs * (avg_probs + 1e-10).log()).sum()).exp()
        
        return quantized, loss, encoding_indices, perplexity

class BrainStateVQVAE(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, latent_dim=256, 
                 n_embeddings=8192):
        """
        VQ-VAE for brain state tokenization
        input_dim: Gabor features dimension
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(n_embeddings, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, vq_loss, tokens, perplexity = self.vq(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        return x_recon, recon_loss, vq_loss, tokens, perplexity
    
    def tokenize(self, x):
        """
        Convert brain state to token
        """
        with torch.no_grad():
            z = self.encoder(x)
            _, _, tokens, _ = self.vq(z)
        return tokens.squeeze().cpu().numpy()
    
    def decode_tokens(self, tokens):
        """
        Decode tokens back to brain states
        """
        with torch.no_grad():
            z_q = self.vq.embedding(tokens)
            x_recon = self.decoder(z_q)
        return x_recon.cpu().numpy()
```

## 4. Cross-Modal Temporal Alignment

### Aligning EEG and WHOOP Data
```python
class CrossModalAligner:
    def __init__(self, eeg_fs=256, whoop_fs=1, delay_range=(5, 10)):
        """
        Align EEG and WHOOP with autonomic delay
        """
        self.eeg_fs = eeg_fs
        self.whoop_fs = whoop_fs
        self.delay_range = delay_range  # seconds
        
    def find_optimal_delay(self, eeg_features, whoop_features, 
                          correlation_window=60):
        """
        Find optimal delay between brain and autonomic signals
        """
        # Downsample EEG features to match WHOOP rate
        eeg_downsampled = signal.resample(
            eeg_features,
            len(whoop_features)
        )
        
        # Test different delays
        delays = np.arange(
            self.delay_range[0],
            self.delay_range[1],
            0.5  # 0.5 second steps
        )
        
        correlations = []
        for delay in delays:
            delay_samples = int(delay)
            
            if delay_samples < len(whoop_features):
                # Shift WHOOP data by delay
                whoop_shifted = whoop_features[delay_samples:]
                eeg_aligned = eeg_downsampled[:len(whoop_shifted)]
                
                # Compute correlation
                if len(eeg_aligned) > correlation_window:
                    corr = np.corrcoef(eeg_aligned, whoop_shifted)[0, 1]
                    correlations.append(corr)
                else:
                    correlations.append(0)
            else:
                correlations.append(0)
        
        # Find optimal delay
        optimal_idx = np.argmax(np.abs(correlations))
        optimal_delay = delays[optimal_idx]
        optimal_corr = correlations[optimal_idx]
        
        return optimal_delay, optimal_corr, delays, correlations
    
    def align_multimodal_data(self, eeg_data, whoop_data, optimal_delay):
        """
        Align data streams with optimal delay
        """
        delay_samples_whoop = int(optimal_delay * self.whoop_fs)
        delay_samples_eeg = int(optimal_delay * self.eeg_fs)
        
        # Align data
        aligned_data = {
            'eeg': eeg_data[:, delay_samples_eeg:],
            'whoop': whoop_data[:-delay_samples_whoop] if delay_samples_whoop > 0 
                     else whoop_data,
            'delay': optimal_delay,
            'timestamps': self.generate_aligned_timestamps(
                eeg_data.shape[1] - delay_samples_eeg,
                self.eeg_fs
            )
        }
        
        return aligned_data
    
    def generate_aligned_timestamps(self, n_samples, fs):
        """
        Generate aligned timestamp array
        """
        return np.arange(n_samples) / fs
```

## 5. Validation Dataset Creation

### Known Thought-Behavior Pairs
```python
class ValidationDatasetBuilder:
    def __init__(self):
        """
        Build validation dataset with ground truth
        """
        self.tasks = {
            'rest': {
                'instruction': 'Close eyes and relax',
                'duration': 120,
                'expected_tokens': 'REST_DEFAULT',
                'validation_metric': 'alpha_power'
            },
            'attention_visual': {
                'instruction': 'Focus on center cross',
                'duration': 60,
                'expected_tokens': 'ATTENTION_HIGH',
                'validation_metric': 'p300_amplitude'
            },
            'working_memory': {
                'instruction': 'n-back task',
                'duration': 300,
                'expected_tokens': 'WM_HIGH',
                'validation_metric': 'frontal_theta'
            },
            'motor_imagery': {
                'instruction': 'Imagine hand movement',
                'duration': 120,
                'expected_tokens': 'MOTOR_PREP',
                'validation_metric': 'beta_erd'
            },
            'emotional_positive': {
                'instruction': 'View positive images',
                'duration': 180,
                'expected_tokens': 'VALENCE_POS',
                'validation_metric': 'frontal_asymmetry'
            }
        }
        
    def create_validation_protocol(self):
        """
        Create standardized validation protocol
        """
        protocol = []
        
        for task_name, task_info in self.tasks.items():
            protocol.append({
                'task': task_name,
                'instruction': task_info['instruction'],
                'duration': task_info['duration'],
                'markers': self.generate_task_markers(task_info),
                'validation': task_info['validation_metric']
            })
        
        return protocol
    
    def generate_task_markers(self, task_info):
        """
        Generate event markers for task
        """
        if 'n-back' in task_info['instruction']:
            # Generate n-back stimuli
            return self.generate_nback_markers(task_info['duration'])
        elif 'Focus' in task_info['instruction']:
            # Visual attention markers
            return self.generate_attention_markers(task_info['duration'])
        else:
            # Simple start/end markers
            return [0, task_info['duration']]
    
    def validate_token_generation(self, tokens, task_name):
        """
        Validate generated tokens match expected patterns
        """
        expected = self.tasks[task_name]['expected_tokens']
        
        # Token category extraction
        token_categories = [self.decode_token_category(t) for t in tokens]
        
        # Compute match percentage
        matches = [cat == expected for cat in token_categories]
        accuracy = np.mean(matches)
        
        # Temporal consistency
        consistency = self.compute_temporal_consistency(token_categories)
        
        return {
            'task': task_name,
            'accuracy': accuracy,
            'consistency': consistency,
            'expected': expected,
            'actual_distribution': np.bincount(tokens) / len(tokens)
        }
    
    def decode_token_category(self, token_id):
        """
        Extract category from token ID
        """
        # Assuming 8192 tokens, 8 categories
        category_size = 8192 // 8
        category_id = token_id // category_size
        
        categories = [
            'ATTENTION', 'REST', 'MEMORY', 'EMOTION',
            'MOTOR', 'CONSCIOUS', 'INTEGRATION', 'TRANSITION'
        ]
        
        return categories[category_id]
    
    def compute_temporal_consistency(self, token_sequence):
        """
        Measure how consistent tokens are over time
        """
        # Count transitions
        transitions = np.sum([
            token_sequence[i] != token_sequence[i+1]
            for i in range(len(token_sequence)-1)
        ])
        
        # Normalize by sequence length
        consistency = 1 - (transitions / len(token_sequence))
        
        return consistency
```

## 6. Integration Test Suite

### Complete Validation Pipeline
```python
class BrainTokenizationValidator:
    def __init__(self):
        """
        Complete validation suite
        """
        self.p300_detector = MinimalP300Detector()
        self.gabor_validator = HolonomicTransformValidator()
        self.aligner = CrossModalAligner()
        self.dataset_builder = ValidationDatasetBuilder()
        
        # Initialize VQ-VAE model
        self.vqvae = BrainStateVQVAE()
        
    def run_validation_suite(self, eeg_data, whoop_data, 
                           behavioral_data, event_markers):
        """
        Run complete validation
        """
        results = {}
        
        # 1. Test P300 consistency
        print("Testing P300 consistency...")
        p300_results, consistency = self.p300_detector.validate_consistency(
            eeg_data, event_markers
        )
        results['p300_consistency'] = consistency
        
        # 2. Optimize Gabor parameters
        print("Optimizing Gabor parameters...")
        optimal_gabor = self.gabor_validator.optimize_gabor_parameters(
            eeg_data[0], behavioral_data
        )
        results['optimal_gabor'] = optimal_gabor
        
        # 3. Test temporal alignment
        print("Testing temporal alignment...")
        eeg_features = self.extract_summary_features(eeg_data[0])
        whoop_features = whoop_data[0][:, 0]  # Use HRV
        
        optimal_delay, correlation, _, _ = self.aligner.find_optimal_delay(
            eeg_features, whoop_features
        )
        results['temporal_alignment'] = {
            'optimal_delay': optimal_delay,
            'correlation': correlation
        }
        
        # 4. Train VQ-VAE
        print("Training VQ-VAE tokenizer...")
        vqvae_results = self.train_vqvae(eeg_data[0])
        results['vqvae'] = vqvae_results
        
        # 5. Generate validation dataset
        print("Creating validation dataset...")
        validation_protocol = self.dataset_builder.create_validation_protocol()
        results['validation_protocol'] = validation_protocol
        
        # 6. Test token consistency
        print("Testing token consistency...")
        token_consistency = self.test_token_consistency(eeg_data)
        results['token_consistency'] = token_consistency
        
        # Summary report
        results['summary'] = self.generate_summary_report(results)
        
        return results
    
    def extract_summary_features(self, eeg_data):
        """
        Extract summary features for alignment
        """
        # Use alpha power as proxy
        alpha_band = signal.filtfilt(
            *signal.butter(4, [8, 13], 'bandpass', fs=256),
            eeg_data,
            axis=1
        )
        
        # Average power across channels
        alpha_power = np.mean(alpha_band**2, axis=0)
        
        return alpha_power
    
    def train_vqvae(self, eeg_data, n_epochs=50):
        """
        Train VQ-VAE on brain states
        """
        # Extract Gabor features
        features = self.gabor_validator.extract_gabor_features(eeg_data)
        
        # Convert to PyTorch
        features_tensor = torch.FloatTensor(features)
        
        # Train VQ-VAE
        optimizer = torch.optim.Adam(self.vqvae.parameters(), lr=1e-3)
        
        losses = []
        perplexities = []
        
        for epoch in range(n_epochs):
            # Forward pass
            x_recon, recon_loss, vq_loss, tokens, perplexity = \
                self.vqvae(features_tensor)
            
            # Total loss
            loss = recon_loss + vq_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            perplexities.append(perplexity.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, "
                      f"Perplexity={perplexity.item():.2f}")
        
        # Test tokenization
        tokens = self.vqvae.tokenize(features_tensor)
        
        return {
            'final_loss': losses[-1],
            'final_perplexity': perplexities[-1],
            'unique_tokens': len(np.unique(tokens)),
            'token_distribution': np.bincount(tokens) / len(tokens)
        }
    
    def test_token_consistency(self, eeg_sessions):
        """
        Test if same thoughts produce same tokens
        """
        # Extract features and tokenize each session
        session_tokens = []
        
        for session in eeg_sessions:
            features = self.gabor_validator.extract_gabor_features(session)
            tokens = self.vqvae.tokenize(torch.FloatTensor(features))
            session_tokens.append(tokens)
        
        # Compare token sequences
        consistency_scores = []
        
        for i in range(len(session_tokens)-1):
            for j in range(i+1, len(session_tokens)):
                # Align sequences (may have different lengths)
                min_len = min(len(session_tokens[i]), len(session_tokens[j]))
                
                # Exact match rate
                exact_match = np.mean(
                    session_tokens[i][:min_len] == session_tokens[j][:min_len]
                )
                
                # Category match rate
                cat_i = [self.dataset_builder.decode_token_category(t) 
                        for t in session_tokens[i][:min_len]]
                cat_j = [self.dataset_builder.decode_token_category(t) 
                        for t in session_tokens[j][:min_len]]
                
                category_match = np.mean([ci == cj for ci, cj in zip(cat_i, cat_j)])
                
                consistency_scores.append({
                    'sessions': (i, j),
                    'exact_match': exact_match,
                    'category_match': category_match
                })
        
        return consistency_scores
    
    def generate_summary_report(self, results):
        """
        Generate summary validation report
        """
        report = {
            'p300_reliability': results['p300_consistency']['overall_score'],
            'optimal_gabor_score': results['optimal_gabor']['score'],
            'temporal_alignment_r': results['temporal_alignment']['correlation'],
            'vqvae_perplexity': results['vqvae']['final_perplexity'],
            'unique_tokens_used': results['vqvae']['unique_tokens'],
            'token_consistency': np.mean([
                s['category_match'] 
                for s in results['token_consistency']
            ])
        }
        
        # Overall validation score
        report['overall_validation_score'] = np.mean([
            report['p300_reliability'],
            report['optimal_gabor_score'],
            abs(report['temporal_alignment_r']),
            report['token_consistency']
        ])
        
        # Recommendations
        report['recommendations'] = []
        
        if report['p300_reliability'] < 0.7:
            report['recommendations'].append(
                "P300 detection needs improvement - check electrode placement"
            )
        
        if report['token_consistency'] < 0.6:
            report['recommendations'].append(
                "Token consistency low - consider more training data"
            )
        
        if report['unique_tokens_used'] < 100:
            report['recommendations'].append(
                "Low token diversity - may need richer input features"
            )
        
        return report
```

## Usage Example

```python
# Initialize validator
validator = BrainTokenizationValidator()

# Load your data
eeg_sessions = load_eeg_data()  # List of sessions
whoop_data = load_whoop_data()  # Corresponding WHOOP data
behavioral_data = load_behavioral_labels()  # Task labels
event_markers = load_event_markers()  # Stimulus timings

# Run validation
results = validator.run_validation_suite(
    eeg_sessions,
    whoop_data,
    behavioral_data,
    event_markers
)

# Print summary
print("\n=== VALIDATION SUMMARY ===")
print(f"P300 Reliability: {results['summary']['p300_reliability']:.2f}")
print(f"Optimal Gabor Score: {results['summary']['optimal_gabor_score']:.2f}")
print(f"Temporal Alignment: r={results['summary']['temporal_alignment_r']:.2f}")
print(f"Token Consistency: {results['summary']['token_consistency']:.2f}")
print(f"Overall Score: {results['summary']['overall_validation_score']:.2f}")

print("\nRecommendations:")
for rec in results['summary']['recommendations']:
    print(f"- {rec}")
```

This validation framework provides a complete pipeline to test the reliability of our brain-to-token conversion system using the minimal preprocessing approach we discovered.