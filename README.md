# Brain-Computer Interface Research

This repository contains comprehensive research on Brain-Computer Interface (BCI) technology, focusing on converting neural activity into computational tokens that can be processed by language models.

## Overview

The research presents a mathematical framework combining:
- **Chaos Theory** - Analyzing neural dynamics as self-organizing systems
- **Holonomic Brain Theory** - Using Gabor transforms for distributed information extraction
- **Global Neuronal Workspace** - Detecting consciousness markers like P300 and gamma synchronization
- **Integrated Information Theory** - Measuring consciousness through information integration

## Key Components

### 1. Signal Processing Pipeline
- EEG acquisition (64+ channels, 256-500Hz)
- Minimal preprocessing (0.5Hz high-pass filter)
- Feature extraction (P300, alpha power, HRV, gamma synchronization)
- Holonomic transformation using Gabor wavelets
- VQ-VAE tokenization to 8192 token vocabulary

### 2. Mathematical Framework
- **Master Equation**: `Token_t = H[S_t · PE_t · Φ_t · GA_t]`
- **Lyapunov Exponent**: For chaos detection
- **Phase Locking Value**: For synchronization analysis
- **Integrated Information**: Using Earth Mover's Distance
- **Sample Entropy**: For complexity measurement

### 3. Implementation
The repository includes Python implementations for:
- Phase space reconstruction
- Attractor analysis
- P300 detection
- Gamma synchronization measurement
- Holonomic feature extraction
- Real-time tokenization

## Hardware Requirements

### Minimum Setup
- **EEG**: 64-channel system, 256 Hz sampling
- **Biometrics**: WHOOP band or similar
- **GPU**: 8GB+ VRAM for VQ-VAE training
- **Storage**: ~100GB for raw data and features

### Supported Hardware
- Neurable MW75 (12 channels, 500Hz)
- OpenBCI systems
- Standard clinical EEG systems

## Software Stack
- Python 3.8+
- NumPy, SciPy, MNE for signal processing
- PyTorch for deep learning
- BrainFlow for hardware interfacing
- Real-time processing with asyncio

## Most Reliable Features
Based on research, these features show highest consistency:
1. P300 amplitude (r = 0.88)
2. Alpha power (r = 0.89)
3. HRV from WHOOP (r = 0.85-0.88)
4. Beta ERD (r = 0.89)
5. Gamma synchrony (r = 0.82)

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Connect your EEG hardware
4. Run the example scripts in the `examples/` directory

## Website
Open `index.html` in a web browser to view the interactive documentation and code examples.

## Contributing
This is open source research. Contributions, validations, and improvements are welcome!

## License
MIT License - See LICENSE file for details

## Contact
For collaboration or questions about the research, please open an issue or submit a pull request.

## Acknowledgments
This work builds on research from:
- Makin et al. (2020) - Brain2Text
- Kostas et al. (2021) - BERT for EEG  
- Willett et al. (2021) - Motor Imagery BCIs
- Moses et al. (2021) - Semantic Thought Decoding