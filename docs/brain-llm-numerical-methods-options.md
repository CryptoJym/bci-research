# Numerical Methods Options for Brain-to-LLM Tokenization

## 1. Signal Processing Methods

### Fast Fourier Transform (FFT) - Frequency Analysis
```python
# Options comparison
methods = {
    'numpy_fft': {
        'pros': ['Built-in', 'Well-tested', 'Good for batch'],
        'cons': ['Not optimized for real-time'],
        'use_case': 'Offline analysis'
    },
    'scipy_fftpack': {
        'pros': ['Optimized C backend', 'Real FFT variants'],
        'cons': ['Still batch-oriented'],
        'use_case': 'Near real-time with buffering'
    },
    'pyfftw': {
        'pros': ['FFTW backend', 'Multi-threaded', 'Real-time capable'],
        'cons': ['External dependency', 'Setup complexity'],
        'use_case': 'Production real-time system'
    }
}

# Real-time FFT implementation
import pyfftw
import numpy as np

class RealtimeFFT:
    def __init__(self, n_samples=256, n_channels=64):
        # Pre-allocate aligned arrays for FFTW
        self.input = pyfftw.empty_aligned(
            (n_channels, n_samples), dtype='float64'
        )
        self.output = pyfftw.empty_aligned(
            (n_channels, n_samples//2 + 1), dtype='complex128'
        )
        
        # Plan FFT (expensive, do once)
        self.fft = pyfftw.FFTW(
            self.input, self.output,
            axes=(1,),
            direction='FFTW_FORWARD',
            flags=('FFTW_MEASURE',),
            threads=4
        )
    
    def process(self, data):
        self.input[:] = data
        self.fft()
        return self.output
```

### Wavelet Transform Options
```python
# Continuous Wavelet Transform (CWT) options
wavelet_methods = {
    'pywt': {
        'implementation': 'PyWavelets',
        'pros': ['Complete wavelet library', 'Many wavelet families'],
        'cons': ['Not optimized for Gabor'],
        'example': '''
import pywt
coeffs, freqs = pywt.cwt(signal, scales, 'cmor1.5-1.0')
        '''
    },
    'custom_gabor': {
        'implementation': 'NumPy/Numba',
        'pros': ['Optimized for our use case', 'GPU ready'],
        'cons': ['Must implement ourselves'],
        'example': '''
@numba.jit(nopython=True, parallel=True)
def gabor_transform(signal, scales, orientations):
    # Custom optimized implementation
    pass
        '''
    },
    'tensorflow_gabor': {
        'implementation': 'TF Signal',
        'pros': ['GPU acceleration', 'Batch processing'],
        'cons': ['Heavy dependency'],
        'example': '''
import tensorflow as tf
gabor_filters = tf.signal.gabor_filterbank(...)
        '''
    }
}
```

### Hilbert Transform for Phase
```python
# Options for instantaneous phase extraction
from scipy.signal import hilbert
from numba import jit

# Standard approach
def hilbert_scipy(signal):
    analytic = hilbert(signal)
    phase = np.angle(analytic)
    amplitude = np.abs(analytic)
    return phase, amplitude

# Optimized for multiple channels
@jit(nopython=True, parallel=True)
def hilbert_numba(signals):
    n_channels, n_samples = signals.shape
    phases = np.zeros_like(signals)
    amplitudes = np.zeros_like(signals)
    
    for ch in numba.prange(n_channels):
        # FFT-based Hilbert
        fft_sig = np.fft.fft(signals[ch])
        fft_sig[1:n_samples//2] *= 2
        fft_sig[n_samples//2+1:] = 0
        analytic = np.fft.ifft(fft_sig)
        
        phases[ch] = np.angle(analytic)
        amplitudes[ch] = np.abs(analytic)
    
    return phases, amplitudes
```

## 2. Matrix Operations & Linear Algebra

### Precision Matrix Computation
```python
# Options for computing precision (inverse covariance)
linear_algebra_methods = {
    'numpy_inv': {
        'method': 'Direct inversion',
        'complexity': 'O(n³)',
        'stability': 'Poor for ill-conditioned',
        'code': 'np.linalg.inv(covariance)'
    },
    'scipy_pinv': {
        'method': 'Pseudo-inverse',
        'complexity': 'O(n³)',
        'stability': 'Better, handles singular',
        'code': 'scipy.linalg.pinv(covariance)'
    },
    'cholesky': {
        'method': 'Cholesky decomposition',
        'complexity': 'O(n³/3)',
        'stability': 'Best for positive definite',
        'code': '''
L = np.linalg.cholesky(covariance)
precision = scipy.linalg.solve_triangular(
    L.T, scipy.linalg.solve_triangular(L, np.eye(n), lower=True)
)
        '''
    },
    'graphical_lasso': {
        'method': 'Sparse inverse covariance',
        'complexity': 'O(n³) but sparse',
        'stability': 'Excellent, adds regularization',
        'code': '''
from sklearn.covariance import GraphicalLassoCV
model = GraphicalLassoCV()
model.fit(data)
precision = model.precision_
        '''
    }
}
```

### Eigendecomposition for Network Analysis
```python
# Fast eigendecomposition options
eigen_methods = {
    'full_eigen': {
        'use': 'All eigenvalues/vectors',
        'method': 'np.linalg.eigh',
        'time': 'O(n³)'
    },
    'sparse_eigen': {
        'use': 'Few largest/smallest',
        'method': 'scipy.sparse.linalg.eigsh',
        'time': 'O(n²k) for k eigenvalues'
    },
    'randomized': {
        'use': 'Approximate top-k',
        'method': 'sklearn.decomposition.PCA',
        'time': 'O(n²k) but faster constant'
    }
}

# Example: Fast Fiedler vector for graph partitioning
from scipy.sparse import csgraph

def fast_fiedler_vector(adjacency_matrix):
    # Compute Laplacian
    laplacian = csgraph.laplacian(adjacency_matrix, normed=True)
    
    # Get second smallest eigenvalue/vector (Fiedler)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        laplacian, k=2, which='SM'
    )
    
    fiedler_vector = eigenvectors[:, 1]
    return fiedler_vector
```

## 3. Optimization Methods

### Earth Mover's Distance (EMD) Optimization
```python
# EMD computation options
emd_methods = {
    'pyemd': {
        'pros': ['Fast C++ implementation', 'Well tested'],
        'cons': ['1D/2D only'],
        'usage': '''
from pyemd import emd
distance = emd(hist1, hist2, distance_matrix)
        '''
    },
    'pot': {
        'pros': ['Python Optimal Transport', 'GPU support', 'Many algorithms'],
        'cons': ['Heavier dependency'],
        'usage': '''
import ot
distance = ot.emd2(hist1, hist2, cost_matrix)
        '''
    },
    'custom_sinkhorn': {
        'pros': ['Differentiable', 'Very fast approximation'],
        'cons': ['Approximate only'],
        'usage': '''
def sinkhorn_distance(p, q, cost, epsilon=0.1, max_iter=100):
    # Entropic regularization
    K = np.exp(-cost / epsilon)
    u = np.ones_like(p)
    for _ in range(max_iter):
        v = q / (K.T @ u)
        u = p / (K @ v)
    return np.sum(u * ((K * cost) @ v))
        '''
    }
}
```

### Codebook Learning (Vector Quantization)
```python
# VQ-VAE codebook optimization options
vq_methods = {
    'kmeans': {
        'method': 'K-means clustering',
        'pros': ['Simple', 'Fast convergence'],
        'cons': ['Local minima', 'Requires good init'],
        'implementation': '''
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=8192, batch_size=1000)
kmeans.fit(brain_states)
codebook = kmeans.cluster_centers_
        '''
    },
    'som': {
        'method': 'Self-Organizing Map',
        'pros': ['Topology preserving', 'Online learning'],
        'cons': ['Slower', 'More parameters'],
        'implementation': '''
from minisom import MiniSom
som = MiniSom(64, 128, brain_state_dim)
som.train(brain_states, num_iterations)
codebook = som.get_weights().reshape(-1, brain_state_dim)
        '''
    },
    'vqvae_gradient': {
        'method': 'Gradient-based VQ-VAE',
        'pros': ['End-to-end learning', 'Optimal for task'],
        'cons': ['Requires differentiable approximation'],
        'implementation': '''
# Straight-through estimator
def vq_loss(inputs, codebook):
    distances = torch.cdist(inputs, codebook)
    indices = distances.argmin(dim=1)
    quantized = codebook[indices]
    
    # Straight-through gradient
    quantized = inputs + (quantized - inputs).detach()
    
    # Commitment loss
    loss = F.mse_loss(inputs.detach(), quantized) + \
           0.25 * F.mse_loss(inputs, quantized.detach())
    
    return quantized, loss, indices
        '''
    }
}
```

## 4. Statistical Filtering

### Kalman Filter Variants
```python
# Kalman filter options for prediction error
kalman_variants = {
    'standard': {
        'assumption': 'Linear, Gaussian',
        'complexity': 'O(n³) per update',
        'use_case': 'Simple state tracking'
    },
    'extended': {
        'assumption': 'Nonlinear, Gaussian',
        'complexity': 'O(n³) + Jacobian',
        'use_case': 'Nonlinear brain dynamics'
    },
    'unscented': {
        'assumption': 'Nonlinear, Gaussian',
        'complexity': 'O(n³) + sigma points',
        'use_case': 'Highly nonlinear, no Jacobian'
    },
    'particle': {
        'assumption': 'Any distribution',
        'complexity': 'O(n×particles)',
        'use_case': 'Non-Gaussian brain states'
    }
}

# Efficient UKF implementation
class UnscentedKalmanFilter:
    def __init__(self, dim_state, dim_obs, dt=1/256):
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.dt = dt
        
        # Sigma point parameters
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 3 - dim_state
        
        # Weights for sigma points
        self.lambda_ = self.alpha**2 * (dim_state + self.kappa) - dim_state
        self.Wm = np.zeros(2 * dim_state + 1)
        self.Wc = np.zeros(2 * dim_state + 1)
        self.Wm[0] = self.lambda_ / (dim_state + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wm[1:] = 1 / (2 * (dim_state + self.lambda_))
        self.Wc[1:] = self.Wm[1:]
        
    def generate_sigma_points(self, x, P):
        """Generate sigma points for UKF"""
        n = len(x)
        sigma_points = np.zeros((2*n + 1, n))
        
        sigma_points[0] = x
        
        # Matrix square root
        try:
            S = np.linalg.cholesky((n + self.lambda_) * P)
        except:
            # Fall back to SVD if not positive definite
            U, s, _ = np.linalg.svd(P)
            S = U @ np.diag(np.sqrt(s))
        
        for i in range(n):
            sigma_points[i+1] = x + S[i]
            sigma_points[n+i+1] = x - S[i]
            
        return sigma_points
```

## 5. Graph Analysis Methods

### Efficient Graph Partitioning
```python
# Methods for finding minimum information partition
partition_methods = {
    'spectral': {
        'method': 'Spectral clustering',
        'complexity': 'O(n³)',
        'quality': 'Good for balanced cuts',
        'implementation': '''
from sklearn.cluster import SpectralClustering
clustering = SpectralClustering(
    n_clusters=2,
    affinity='precomputed',
    assign_labels='kmeans'
)
partition = clustering.fit_predict(adjacency_matrix)
        '''
    },
    'metis': {
        'method': 'METIS multilevel',
        'complexity': 'O(n log n)',
        'quality': 'Very fast, good quality',
        'implementation': '''
import pymetis
# Requires graph in CSR format
n_cuts, partition = pymetis.part_graph(
    2, adjacency=adjacency_lists
)
        '''
    },
    'karger': {
        'method': 'Randomized min-cut',
        'complexity': 'O(n² log³ n)',
        'quality': 'Probabilistic guarantee',
        'implementation': '''
def karger_min_cut(graph, iterations=100):
    min_cut = float('inf')
    best_partition = None
    
    for _ in range(iterations):
        g = graph.copy()
        while g.number_of_nodes() > 2:
            # Random edge contraction
            edge = random.choice(list(g.edges()))
            g = nx.contracted_edge(g, edge, self_loops=False)
        
        if g.number_of_edges() < min_cut:
            min_cut = g.number_of_edges()
            best_partition = list(g.nodes())
    
    return min_cut, best_partition
        '''
    }
}
```

### Network Metrics Computation
```python
# Efficient computation of graph metrics
import networkx as nx
from joblib import Parallel, delayed

def compute_network_metrics(adjacency_matrix, metrics=['efficiency', 'clustering', 'small_world']):
    """Compute various network metrics efficiently"""
    
    G = nx.from_numpy_array(adjacency_matrix)
    results = {}
    
    if 'efficiency' in metrics:
        # Global efficiency
        results['global_efficiency'] = nx.global_efficiency(G)
        
        # Local efficiency (parallelized)
        def local_eff(node):
            return nx.local_efficiency(G, node)
        
        local_effs = Parallel(n_jobs=-1)(
            delayed(local_eff)(node) for node in G.nodes()
        )
        results['mean_local_efficiency'] = np.mean(local_effs)
    
    if 'clustering' in metrics:
        results['clustering_coefficient'] = nx.average_clustering(G)
    
    if 'small_world' in metrics:
        # Compute small-world coefficient
        # σ = (C/C_rand) / (L/L_rand)
        n = len(G)
        m = G.number_of_edges()
        
        # Random graph with same n, m
        G_rand = nx.gnm_random_graph(n, m)
        
        C = nx.average_clustering(G)
        C_rand = nx.average_clustering(G_rand)
        
        L = nx.average_shortest_path_length(G)
        L_rand = nx.average_shortest_path_length(G_rand)
        
        results['small_world_coefficient'] = (C / C_rand) / (L / L_rand)
    
    return results
```

## 6. Real-Time Processing Pipeline

### Stream Processing Architecture
```python
import asyncio
from collections import deque
import numpy as np

class StreamProcessor:
    def __init__(self, window_size=256, overlap=128):
        self.window_size = window_size
        self.overlap = overlap
        self.buffer = deque(maxlen=window_size)
        
        # Pre-allocate processing arrays
        self.window = np.zeros(window_size)
        self.fft_output = np.zeros(window_size // 2 + 1, dtype=complex)
        
        # Initialize processing components
        self.init_processors()
        
    def init_processors(self):
        """Initialize all numerical processors"""
        # FFT planner
        self.fft_plan = pyfftw.FFTW(
            self.window, self.fft_output,
            flags=('FFTW_MEASURE',)
        )
        
        # Filter banks
        self.filters = self.design_filter_bank()
        
        # Feature extractors
        self.feature_extractors = {
            'spectral': self.extract_spectral_features,
            'temporal': self.extract_temporal_features,
            'connectivity': self.extract_connectivity_features
        }
    
    async def process_stream(self, data_stream):
        """Async processing of data stream"""
        async for sample in data_stream:
            self.buffer.append(sample)
            
            if len(self.buffer) >= self.window_size:
                # Process window
                result = await self.process_window()
                
                # Slide window
                for _ in range(self.overlap):
                    self.buffer.popleft()
                
                yield result
    
    async def process_window(self):
        """Process single window of data"""
        # Copy to pre-allocated array
        self.window[:] = self.buffer
        
        # Parallel feature extraction
        features = await asyncio.gather(
            self.extract_features_async('spectral'),
            self.extract_features_async('temporal'),
            self.extract_features_async('connectivity')
        )
        
        return self.combine_features(features)
```

## Performance Optimization Summary

### Method Selection Guide

| Component | Development | Production | GPU-Accelerated |
|-----------|-------------|------------|-----------------|
| FFT | NumPy | pyFFTW | CuPy/TensorFlow |
| Wavelets | PyWavelets | Custom Numba | PyTorch |
| Matrix Ops | NumPy | BLAS/LAPACK | CuBLAS |
| EMD | pyemd | POT | POT-GPU |
| Kalman | filterpy | Custom Numba | CuPy |
| Graph | NetworkX | graph-tool | cuGraph |
| VQ | scikit-learn | Faiss | Faiss-GPU |

### Computational Complexity

| Operation | Complexity | Frequency | Optimization Priority |
|-----------|------------|-----------|---------------------|
| FFT | O(n log n) | Every sample | HIGH |
| Phase computation | O(n) | Every sample | HIGH |
| Connectivity | O(n²) | Every window | MEDIUM |
| EMD/Phi | O(n³) | Every second | LOW (approximate) |
| Codebook search | O(k) | Every token | HIGH |

This comprehensive set of numerical methods provides multiple implementation options optimized for different scenarios in the brain-to-LLM tokenization pipeline.