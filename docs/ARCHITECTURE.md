# PinyByteCNN Architecture

## Overview

PinyByteCNN implements a Convolutional Neural Network architecture optimized for byte-level text classification in pure Python.

## Core Architecture

### Input Processing

Text inputs are processed through UTF-8 byte encoding:

```
"Hello" → [72, 101, 108, 108, 111] → padded/truncated to max_len
```

### Model Pipeline

1. **Embedding Layer**
   - Maps bytes (0-255) to dense vectors
   - Shape: [vocab_size=256, embed_dim]
   - Learnable parameters: 256 × embed_dim

2. **Convolutional Layers**
   - 1D convolution over byte sequences
   - Kernel sizes: typically 3-5
   - Padding: "same" to preserve sequence length
   - Activation: ReLU

3. **Global Pooling**
   - Combines average and max pooling: (avg + max) / 2
   - Reduces sequence to fixed-size representation
   - Shape: [seq_len, filters] → [filters]

4. **Classification Head**
   - Dense layer with ReLU activation
   - Output layer with sigmoid activation
   - Binary classification: toxicity score [0, 1]

## Multi-Layer Configurations

### Single Layer (ByteCNN)
```
Input[512] → Embed[512,14] → Conv1D[512,28] → Pool[28] → Dense[48] → Output[1]
```

### Multi-Layer (MultiLayerByteCNN)
```
Input[512] → Embed[512,14] → Conv1D[512,28] → Conv1D[512,40] → Pool[40] → Dense[128] → Output[1]
```

## Optimization Strategies

### Memory Efficiency

1. **Buffer Reuse**: Pre-allocated arrays reused across predictions
2. **In-place Operations**: Minimize memory allocations
3. **Fused Operations**: BatchNorm folded into convolution weights

### Performance Optimizations

1. **Vectorized Operations**: List comprehensions for batch processing
2. **Early Termination**: Skip computations for padded regions
3. **Weight Quantization**: Optional for further compression

## Model Variants

### Production Models

| Model | Parameters | Accuracy | Use Case |
|-------|------------|----------|----------|
| ByteCNN-10K | 10,009 | 78.97% | Ultra-lightweight edge |
| ByteCNN-15K | 15,360 | 80.23% | Balanced performance |
| ByteCNN-32K | 32,768 | 82.15% | High accuracy |

### Architecture Specifications

#### ByteCNN-10K (Ultra-Lightweight)
- Embedding: 256 × 12 = 3,072 params
- Conv1D: (12 → 40) × 3 + 40 bias = 1,480 params
- Dense: (40 → 128) × 1 + 128 bias = 5,248 params
- Output: 128 × 1 + 1 bias = 129 params
- **Total: 10,009 parameters**

## Training Considerations

### Data Requirements

- Balanced toxic/non-toxic examples
- UTF-8 encoded text samples
- Maximum sequence length: 512 bytes typically
- Minimum training set: 10K balanced samples

### Training Pipeline

1. **Preprocessing**: Text → UTF-8 bytes → padding/truncation
2. **Data Augmentation**: Character-level noise, truncation variations  
3. **Loss Function**: Binary cross-entropy
4. **Optimization**: Adam optimizer, learning rate scheduling
5. **Regularization**: Dropout in training, BatchNorm for stability

## Inference Strategies

### Truncate Strategy
- Use first max_len bytes only
- Fastest inference
- May lose information for long texts

### Average Strategy
- Sliding window approach
- Average predictions over multiple windows
- Better coverage for long texts

### Attention Strategy
- Weighted average based on attention scores
- Most accurate for variable-length inputs
- Highest computational cost

## Implementation Details

### Pure Python Constraints

All operations implemented without external dependencies:
- Matrix multiplication: nested list comprehensions
- Convolution: manual kernel application
- Activations: math library functions
- No NumPy, PyTorch, or TensorFlow dependencies

### Memory Layout

```python
# Typical memory usage for ByteCNN-10K
embedding_weights: 256 × 12 × 4 bytes = 12.3 KB
conv_weights: 40 × 12 × 3 × 4 bytes = 5.8 KB  
dense_weights: 128 × 40 × 4 bytes = 20.5 KB
temp_buffers: ~10 KB during inference
# Total: ~50 KB peak memory
```

## Deployment Architecture

### Edge Deployment
- Single file deployment possible
- No runtime dependencies
- Sub-second cold starts
- Minimal resource requirements

### Scalability
- Stateless inference
- Thread-safe operations
- Horizontal scaling ready
- Load balancer compatible