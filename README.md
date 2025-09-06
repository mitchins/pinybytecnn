# PinyByteCNN

Pure Python implementation of ByteCNN for toxicity detection and edge deployment.

## Overview

PinyByteCNN is a lightweight, dependency-free neural network implementation designed for production deployment in constrained environments. It provides CNN-based text classification with minimal memory footprint and fast inference.

## Quick Start

```python
from tinybytecnn.model import ByteCNN

# Create model
model = ByteCNN(
    vocab_size=256,
    embed_dim=14,
    conv_filters=28,
    conv_kernel_size=3,
    hidden_dim=48,
    max_len=512
)

# Predict toxicity
score = model.predict("Hello world")  # Returns float [0.0, 1.0]
```

## Features

- **Pure Python**: No external dependencies beyond standard library
- **Memory Efficient**: Optimized for minimal RAM usage
- **Fast Inference**: Single-pass prediction with pre-allocated buffers  
- **Multiple Architectures**: Support for 1-3 layer CNN configurations
- **Flexible Input**: Handles variable-length text with multiple strategies

## Architecture

ByteCNN processes text through the following pipeline:

1. **Byte Encoding**: Convert text to UTF-8 bytes (0-255)
2. **Embedding**: Map bytes to dense vectors
3. **Convolution**: 1D CNN with ReLU activation
4. **Pooling**: Global average/max pooling
5. **Classification**: Dense layers with sigmoid output

## Installation

Clone the repository and import directly:

```bash
git clone <repository-url>
cd PinyByteCNN
python3 -c "from tinybytecnn.model import ByteCNN; print('Success')"
```

## Usage

### Basic Classification

```python
from tinybytecnn.model import ByteCNN

model = ByteCNN(vocab_size=256, embed_dim=14, conv_filters=28, 
                conv_kernel_size=3, hidden_dim=48)

# Single prediction
score = model.predict("This is a test message")

# Batch processing
texts = ["Hello", "Goodbye", "Test message"]
scores = [model.predict(text) for text in texts]
```

### Multi-Layer Models

```python
from tinybytecnn.multi_layer_optimized import MultiLayerByteCNN

# Define layer configuration
layers = [
    {"in_channels": 14, "out_channels": 28, "kernel_size": 3},
    {"in_channels": 28, "out_channels": 40, "kernel_size": 3}
]

model = MultiLayerByteCNN(layers_config=layers, hidden_dim=128, max_len=512)
score = model.predict("Multi-layer processing")
```

### Prediction Strategies

- `truncate`: Use first max_len bytes (fastest)
- `average`: Average predictions over sliding windows
- `attention`: Weighted average with attention mechanism

```python
score = model.predict("Long text...", strategy="average")
```

## Testing

Run the test suite:

```bash
python3 -m unittest discover tests/
```

### Smoke Tests

Validate against production models:

```bash
python3 tests/test_bytecnn_10k_smoke.py
```

## Performance

| Model | Parameters | Accuracy | Inference Time |
|-------|------------|----------|----------------|
| ByteCNN-10K | 10,009 | 78.97% | 0.5ms |
| ByteCNN-32K | 32,768 | 82.15% | 1.2ms |

*Benchmarks on MacBook Pro M1, single-threaded*

## Production Deployment

PinyByteCNN is designed for edge deployment scenarios:

- **Cloudflare Workers**: Sub-10ms inference
- **AWS Lambda**: Cold start friendly
- **Mobile/IoT**: Minimal memory footprint
- **Air-gapped Systems**: No external dependencies

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment guides.

## Model Architecture Details

For detailed architecture information and training procedures, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Contributing

1. Ensure all tests pass
2. Maintain 80%+ code coverage
3. Follow existing code style
4. Add tests for new features

## License

MIT License - see LICENSE file for details.