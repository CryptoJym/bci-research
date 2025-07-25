#!/usr/bin/env python3
"""
Neurable Quick Implementation Guide
Practical code for consciousness tokenization with MW75
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt

class NeurableConsciousnessAnalyzer:
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate
        
    def quick_analysis(self, eeg_data):
        """
        Run all key analyses on EEG data
        Returns dictionary of numerical markers
        """
        results = {}
        
        # 1. Chaos Analysis
        print("Running chaos analysis...")
        phase_space = self.phase_space_reconstruction(eeg_data)
        results['lyapunov_exponent'] = self.calculate_lyapunov(phase_space)
        results['is_chaotic'] = results['lyapunov_exponent'] > 0
        
        # 2. Inflection Points (like NeuralOptimal)
        print("Detecting inflection points...")
        inflection_points = self.find_inflection_points(eeg_data)
        results['inflection_count'] = len(inflection_points)
        results['inflection_rate'] = len(inflection_points) / (len(eeg_data) / self.fs)
        
        # 3. Attractor Analysis
        print("Analyzing attractors...")
        attractor_info = self.analyze_attractor(phase_space)
        results['attractor_dimension'] = attractor_info['dimension']
        results['is_strange_attractor'] = attractor_info['is_strange']
        
        # 4. Holonomic Features (Pribram)
        print("Extracting holonomic features...")
        gabor_features = self.gabor_analysis(eeg_data)
        results['information_density'] = gabor_features['info_density']
        results['phase_coherence_mean'] = gabor_features['coherence_mean']
        
        # 5. Dehaene Markers
        print("Detecting consciousness markers...")
        p300_info = self.detect_p300(eeg_data)
        results['has_p300'] = p300_info['detected']
        results['p300_latency'] = p300_info['latency']
        
        gamma_sync = self.gamma_synchronization(eeg_data)
        results['gamma_power'] = gamma_sync['power']
        results['is_conscious_state'] = gamma_sync['power'] > 0.3
        
        # 6. Complexity Measures
        print("Computing complexity measures...")
        results['sample_entropy'] = self.sample_entropy(eeg_data)
        results['spectral_entropy'] = self.spectral_entropy(eeg_data)
        
        return results
    
    def phase_space_reconstruction(self, signal, dim=3, tau=10):
        """Takens embedding"""
        N = len(signal)
        M = N - (dim - 1) * tau
        phase_space = np.zeros((M, dim))
        
        for i in range(M):
            for j in range(dim):
                phase_space[i, j] = signal[i + j * tau]
                
        return phase_space
    
    def calculate_lyapunov(self, phase_space):
        """Simplified largest Lyapunov exponent"""
        divergence_rates = []
        
        for i in range(len(phase_space) - 11):
            # Find nearest neighbor
            distances = np.linalg.norm(phase_space - phase_space[i], axis=1)
            distances[i] = np.inf
            nn_idx = np.argmin(distances)
            
            if nn_idx < len(phase_space) - 10:
                initial_sep = distances[nn_idx]
                final_sep = np.linalg.norm(phase_space[i+10] - phase_space[nn_idx+10])
                
                if initial_sep > 0 and final_sep > 0:
                    divergence = np.log(final_sep / initial_sep) / (10 * self.dt)
                    divergence_rates.append(divergence)
        
        return np.mean(divergence_rates) if divergence_rates else 0
    
    def find_inflection_points(self, signal, window=250):
        """Detect state transitions"""
        variance_series = []
        
        for i in range(0, len(signal) - window, window//2):
            variance_series.append(np.var(signal[i:i+window]))
        
        variance_diff = np.diff(variance_series)
        threshold = 2 * np.std(variance_diff)
        inflection_indices = np.where(np.abs(variance_diff) > threshold)[0]
        
        return inflection_indices * (window // 2)
    
    def analyze_attractor(self, phase_space):
        """Estimate fractal dimension"""
        # Simplified correlation dimension
        distances = []
        sample_size = min(1000, len(phase_space))
        indices = np.random.choice(len(phase_space), sample_size, replace=False)
        
        for i in indices:
            for j in indices:
                if i != j:
                    distances.append(np.linalg.norm(phase_space[i] - phase_space[j]))
        
        distances = np.array(distances)
        
        # Count correlations at different scales
        rs = np.logspace(-2, 0, 10)
        correlations = []
        
        for r in rs:
            correlations.append(np.sum(distances < r) / len(distances))
        
        # Estimate dimension from slope
        valid = [(r, c) for r, c in zip(rs, correlations) if c > 0]
        if len(valid) > 2:
            log_r = np.log([r for r, c in valid])
            log_c = np.log([c for r, c in valid])
            dimension, _ = np.polyfit(log_r, log_c, 1)
        else:
            dimension = 0
        
        return {
            'dimension': dimension,
            'is_strange': 0.1 < (dimension % 1) < 0.9  # Non-integer dimension
        }
    
    def gabor_analysis(self, signal, n_frequencies=20):
        """Simplified Gabor transform for holonomic features"""
        frequencies = np.linspace(1, 60, n_frequencies)
        window_size = 128
        
        # Create Gabor wavelets
        coherence_matrix = np.zeros((n_frequencies, n_frequencies))
        
        for i, f1 in enumerate(frequencies):
            for j, f2 in enumerate(frequencies):
                if i < j:
                    # Simple coherence calculation
                    b1, a1 = butter(2, [f1-0.5, f1+0.5], btype='band', fs=self.fs)
                    b2, a2 = butter(2, [f2-0.5, f2+0.5], btype='band', fs=self.fs)
                    
                    try:
                        sig1 = filtfilt(b1, a1, signal)
                        sig2 = filtfilt(b2, a2, signal)
                        
                        phase1 = np.angle(hilbert(sig1))
                        phase2 = np.angle(hilbert(sig2))
                        
                        coherence = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
                        coherence_matrix[i, j] = coherence
                    except:
                        pass
        
        return {
            'info_density': -np.sum(coherence_matrix * np.log(coherence_matrix + 1e-10)),
            'coherence_mean': np.mean(coherence_matrix[coherence_matrix > 0])
        }
    
    def detect_p300(self, signal):
        """Detect P300 events"""
        # Bandpass filter for P300
        b, a = butter(4, [0.1, 30], btype='band', fs=self.fs)
        filtered = filtfilt(b, a, signal)
        
        # Look for positive peaks in 250-500ms windows
        window_start = int(0.25 * self.fs)  # 250ms
        window_end = int(0.5 * self.fs)     # 500ms
        
        detected = False
        latency = None
        
        # Simplified detection
        baseline = np.mean(filtered[:window_start])
        std_baseline = np.std(filtered[:window_start])
        
        for i in range(window_start, min(window_end, len(filtered))):
            if filtered[i] > baseline + 2 * std_baseline:
                detected = True
                latency = i / self.fs * 1000  # Convert to ms
                break
        
        return {'detected': detected, 'latency': latency}
    
    def gamma_synchronization(self, signal):
        """Analyze gamma band power"""
        # Filter in gamma range
        b, a = butter(4, [40, 80], btype='band', fs=self.fs)
        gamma_filtered = filtfilt(b, a, signal)
        
        # Calculate power
        power = np.mean(gamma_filtered ** 2)
        normalized_power = power / np.mean(signal ** 2)
        
        return {'power': normalized_power}
    
    def sample_entropy(self, signal, m=2, r=0.2):
        """Complexity measure"""
        N = len(signal)
        r = r * np.std(signal)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = [signal[i:i+m].tolist() for i in range(N-m+1)]
            C = 0
            
            for i in range(N-m+1):
                matches = 0
                for j in range(N-m+1):
                    if i != j and _maxdist(patterns[i], patterns[j]) <= r:
                        matches += 1
                if matches > 0:
                    C += np.log(matches / (N-m))
            
            return C / (N-m+1)
        
        try:
            return _phi(m) - _phi(m+1)
        except:
            return 0
    
    def spectral_entropy(self, signal):
        """Frequency domain entropy"""
        freqs, psd = signal.periodogram(signal, fs=self.fs)
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        # Calculate entropy
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
        
        return entropy


# Example usage
if __name__ == "__main__":
    # Simulate some EEG data (replace with real Neurable data)
    duration = 10  # seconds
    fs = 500
    t = np.linspace(0, duration, duration * fs)
    
    # Create synthetic EEG with multiple components
    eeg = (np.sin(2 * np.pi * 10 * t) +  # Alpha
           0.5 * np.sin(2 * np.pi * 25 * t) +  # Beta
           0.3 * np.sin(2 * np.pi * 50 * t) +  # Gamma
           0.2 * np.random.randn(len(t)))  # Noise
    
    # Run analysis
    analyzer = NeurableConsciousnessAnalyzer()
    results = analyzer.quick_analysis(eeg)
    
    # Print results
    print("\n=== Consciousness Analysis Results ===")
    print(f"Chaotic dynamics: {'Yes' if results['is_chaotic'] else 'No'}")
    print(f"Lyapunov exponent: {results['lyapunov_exponent']:.3f}")
    print(f"Attractor dimension: {results['attractor_dimension']:.2f}")
    print(f"Strange attractor: {'Yes' if results['is_strange_attractor'] else 'No'}")
    print(f"Inflection points/sec: {results['inflection_rate']:.2f}")
    print(f"Information density: {results['information_density']:.2f}")
    print(f"P300 detected: {'Yes' if results['has_p300'] else 'No'}")
    print(f"Gamma power: {results['gamma_power']:.3f}")
    print(f"Conscious state: {'Yes' if results['is_conscious_state'] else 'No'}")
    print(f"Sample entropy: {results['sample_entropy']:.3f}")
    
    # Generate token
    token = {
        'timestamp': 0,
        'chaos_signature': results['lyapunov_exponent'],
        'attractor_type': 'strange' if results['is_strange_attractor'] else 'regular',
        'complexity': results['sample_entropy'],
        'consciousness_marker': results['is_conscious_state'],
        'information_content': results['information_density']
    }
    
    print("\n=== Generated Token ===")
    print(token)