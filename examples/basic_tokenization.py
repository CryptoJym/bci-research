#!/usr/bin/env python3
"""
Basic BCI Tokenization Example
Demonstrates the core concepts of converting EEG to tokens
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class BasicBCITokenizer:
    def __init__(self, sampling_rate=256, window_size_ms=300):
        self.fs = sampling_rate
        self.window_size = int(window_size_ms * sampling_rate / 1000)
        self.token_vocabulary = self._initialize_vocabulary()
        
    def _initialize_vocabulary(self):
        """Create 8192 token vocabulary (13 bits)"""
        # Bits 0-2: Category (8 categories)
        categories = ['ATTENTION', 'REST', 'MEMORY', 'EMOTION', 
                     'MOTOR', 'CONSCIOUS', 'INTEGRATION', 'TRANSITION']
        
        # Bits 3-12: Specific state within category (1024 states)
        vocabulary = {}
        token_id = 0
        
        for cat_id, category in enumerate(categories):
            for state_id in range(1024):
                vocabulary[token_id] = {
                    'category': category,
                    'category_id': cat_id,
                    'state_id': state_id,
                    'token': f"{category}_{state_id:04d}"
                }
                token_id += 1
                
        return vocabulary
    
    def extract_features(self, eeg_window):
        """Extract key features from EEG window"""
        features = {}
        
        # 1. Band Powers
        features['delta'] = self._band_power(eeg_window, 0.5, 4)
        features['theta'] = self._band_power(eeg_window, 4, 8)
        features['alpha'] = self._band_power(eeg_window, 8, 13)
        features['beta'] = self._band_power(eeg_window, 13, 30)
        features['gamma'] = self._band_power(eeg_window, 30, 100)
        
        # 2. Alpha/Theta Ratio (attention marker)
        features['alpha_theta_ratio'] = features['alpha'] / (features['theta'] + 1e-10)
        
        # 3. Complexity (Sample Entropy)
        features['complexity'] = self._sample_entropy(eeg_window)
        
        # 4. P300-like detection (simplified)
        features['has_p300'] = self._detect_p300_simple(eeg_window)
        
        return features
    
    def _band_power(self, signal, low_freq, high_freq):
        """Calculate power in specific frequency band"""
        nyquist = self.fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if high > 1:
            high = 0.99
            
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        power = np.mean(filtered ** 2)
        
        return power
    
    def _sample_entropy(self, signal, m=2, r=0.2):
        """Calculate sample entropy (complexity measure)"""
        N = len(signal)
        r = r * np.std(signal)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = [signal[i:i+m].tolist() for i in range(N-m+1)]
            C = 0
            
            for i in range(N-m+1):
                matches = sum(1 for j in range(N-m+1) 
                            if i != j and _maxdist(patterns[i], patterns[j]) <= r)
                if matches > 0:
                    C += np.log(matches / (N-m))
                    
            return C / (N-m+1) if N-m+1 > 0 else 0
        
        try:
            return _phi(m) - _phi(m+1)
        except:
            return 0
    
    def _detect_p300_simple(self, signal):
        """Simplified P300 detection"""
        # Look for positive deflection in 250-500ms range
        window_start = int(0.25 * self.fs)
        window_end = int(0.5 * self.fs)
        
        if len(signal) < window_end:
            return False
            
        baseline = np.mean(signal[:window_start])
        peak_amplitude = np.max(signal[window_start:window_end])
        
        return peak_amplitude > baseline + 2 * np.std(signal[:window_start])
    
    def features_to_token(self, features):
        """Map features to token ID"""
        # Determine category based on dominant features
        if features['has_p300']:
            category_id = 5  # CONSCIOUS
        elif features['alpha_theta_ratio'] > 2:
            category_id = 0  # ATTENTION
        elif features['theta'] > features['alpha']:
            category_id = 1  # REST
        elif features['gamma'] > np.mean([features['alpha'], features['beta']]):
            category_id = 2  # MEMORY
        elif features['beta'] > features['alpha']:
            category_id = 4  # MOTOR
        else:
            category_id = 7  # TRANSITION
        
        # Determine state within category using complexity and band powers
        state_features = [
            features['complexity'],
            features['alpha'] / (np.sum([features[b] for b in ['delta', 'theta', 'alpha', 'beta', 'gamma']]) + 1e-10),
            features['beta'] / (features['alpha'] + 1e-10),
            features['gamma'] / (features['beta'] + 1e-10)
        ]
        
        # Quantize to 1024 states
        state_id = int(np.sum(state_features) * 256) % 1024
        
        # Calculate token ID
        token_id = category_id * 1024 + state_id
        
        return token_id, self.token_vocabulary[token_id]
    
    def tokenize_continuous(self, eeg_data, overlap=0.5):
        """Tokenize continuous EEG data"""
        tokens = []
        hop = int(self.window_size * (1 - overlap))
        
        for i in range(0, len(eeg_data) - self.window_size, hop):
            window = eeg_data[i:i + self.window_size]
            features = self.extract_features(window)
            token_id, token_info = self.features_to_token(features)
            
            tokens.append({
                'timestamp': i / self.fs,
                'token_id': token_id,
                'token': token_info['token'],
                'category': token_info['category'],
                'features': features
            })
            
        return tokens
    
    def visualize_tokenization(self, eeg_data, tokens):
        """Visualize EEG data and resulting tokens"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Time axis
        time = np.arange(len(eeg_data)) / self.fs
        
        # Plot EEG
        ax1.plot(time, eeg_data, 'b-', alpha=0.7)
        ax1.set_ylabel('EEG (μV)')
        ax1.set_title('Raw EEG Signal')
        ax1.grid(True, alpha=0.3)
        
        # Plot token categories
        token_times = [t['timestamp'] for t in tokens]
        token_categories = [t['category'] for t in tokens]
        category_ids = [['ATTENTION', 'REST', 'MEMORY', 'EMOTION', 
                        'MOTOR', 'CONSCIOUS', 'INTEGRATION', 'TRANSITION'].index(cat) 
                       for cat in token_categories]
        
        ax2.scatter(token_times, category_ids, c=category_ids, cmap='tab10', s=100)
        ax2.set_ylabel('Category')
        ax2.set_yticks(range(8))
        ax2.set_yticklabels(['ATT', 'REST', 'MEM', 'EMO', 'MOT', 'CON', 'INT', 'TRANS'])
        ax2.set_title('Token Categories')
        ax2.grid(True, alpha=0.3)
        
        # Plot complexity over time
        complexities = [t['features']['complexity'] for t in tokens]
        ax3.plot(token_times, complexities, 'g-', marker='o')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Complexity')
        ax3.set_title('Signal Complexity (Sample Entropy)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Parameters
    duration = 30  # seconds
    fs = 256  # Hz
    
    # Generate synthetic EEG data
    print("Generating synthetic EEG data...")
    t = np.linspace(0, duration, duration * fs)
    
    # Simulate different brain states
    eeg = np.zeros_like(t)
    
    # 0-10s: Rest state (high alpha)
    eeg[:10*fs] = (10 * np.sin(2 * np.pi * 10 * t[:10*fs]) +  # Alpha
                   5 * np.sin(2 * np.pi * 6 * t[:10*fs]) +   # Theta
                   2 * np.random.randn(10*fs))
    
    # 10-20s: Attention state (low alpha, high beta)
    eeg[10*fs:20*fs] = (3 * np.sin(2 * np.pi * 10 * t[:10*fs]) +   # Alpha
                        8 * np.sin(2 * np.pi * 20 * t[:10*fs]) +   # Beta
                        3 * np.random.randn(10*fs))
    
    # 20-30s: Memory task (high gamma)
    eeg[20*fs:] = (5 * np.sin(2 * np.pi * 10 * t[:10*fs]) +   # Alpha
                   6 * np.sin(2 * np.pi * 25 * t[:10*fs]) +   # Beta
                   4 * np.sin(2 * np.pi * 50 * t[:10*fs]) +   # Gamma
                   2 * np.random.randn(10*fs))
    
    # Add some P300-like events
    for event_time in [5, 15, 25]:
        event_sample = int(event_time * fs)
        p300_sample = event_sample + int(0.3 * fs)  # 300ms after event
        if p300_sample < len(eeg):
            eeg[p300_sample:p300_sample+int(0.2*fs)] += 15 * np.exp(-np.linspace(0, 5, int(0.2*fs)))
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BasicBCITokenizer(sampling_rate=fs)
    
    # Tokenize the EEG data
    print("Tokenizing EEG data...")
    tokens = tokenizer.tokenize_continuous(eeg)
    
    # Print token statistics
    print(f"\nTokenization complete!")
    print(f"Total tokens generated: {len(tokens)}")
    print(f"Token rate: {len(tokens) / duration:.2f} tokens/second")
    
    # Category distribution
    categories = [t['category'] for t in tokens]
    unique_categories = list(set(categories))
    print("\nCategory distribution:")
    for cat in unique_categories:
        count = categories.count(cat)
        percentage = count / len(categories) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    # Print first 10 tokens
    print("\nFirst 10 tokens:")
    for i, token in enumerate(tokens[:10]):
        print(f"  {i}: {token['token']} at {token['timestamp']:.2f}s")
    
    # Visualize results
    print("\nGenerating visualization...")
    tokenizer.visualize_tokenization(eeg, tokens)
    
    # Save tokens to file
    import json
    with open('tokens_output.json', 'w') as f:
        json.dump(tokens, f, indent=2)
    print("\nTokens saved to tokens_output.json")