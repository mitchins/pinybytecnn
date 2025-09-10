# PinyByteCNN Scripts

## export_torch_model.py

Universal export script for converting PyTorch ByteCNN models to PinyByteCNN formats.

### Features

- **Dual format support**: Export to both JSON and static Python
- **BatchNorm folding**: Automatically folds BatchNorm into Conv layers
- **Architecture detection**: Automatically detects model structure
- **Metadata preservation**: Includes model info in exports

### Usage

#### Export to both JSON and Python (default)
```bash
python scripts/export_torch_model.py model.pth
```

#### Export to JSON only
```bash
python scripts/export_torch_model.py model.pth --format json
```

#### Export to static Python only
```bash
python scripts/export_torch_model.py model.pth --format python
```

#### With metadata
```bash
python scripts/export_torch_model.py bytecnn_10k.pth \
    --name "ByteCNN-10K" \
    --version "1.0.0" \
    --accuracy 0.7897 \
    --output-dir exports/
```

### Output Formats

#### JSON Format
```json
{
  "model_info": {
    "name": "ByteCNN-10K",
    "version": "1.0.0",
    "parameters": 10009,
    "accuracy": 0.7897
  },
  "weights": {
    "embedding": [...],
    "conv1_weight": [...],
    "conv1_bias": [...],
    "classifier_weight": [...],
    "classifier_bias": [...],
    "output_weight": [...],
    "output_bias": [...]
  },
  "architecture": {
    "vocab_size": 256,
    "embed_dim": 12,
    "conv_filters": 40,
    "kernel_size": 3,
    "dense_dim": 128
  }
}
```

#### Static Python Format
```python
# Model configuration
MODEL_INFO = {
    "name": "ByteCNN-10K",
    "version": "1.0.0",
    "parameters": 10009,
    "exported": "2025-01-06T16:30:00"
}

# Embedding weights [vocab_size, embed_dim]
EMBEDDING_WEIGHT = [
    [...],
    ...
]

# Conv1 weights - BatchNorm folded
CONV1_WEIGHT = [...]
CONV1_BIAS = [...]

# Classifier weights
CLASSIFIER_WEIGHT = [...]
CLASSIFIER_BIAS = [...]

# Output weights
OUTPUT_WEIGHT = [...]
OUTPUT_BIAS = [...]
```

### Supported Model Architectures

The script automatically handles:
- Single-layer ByteCNN models
- Multi-layer models with 2-3 conv layers
- Models with or without BatchNorm
- Various naming conventions (fc1/fc2 or classifier/output)

### BatchNorm Folding

When a model contains BatchNorm layers, the script automatically:
1. Detects BatchNorm layers following Conv layers
2. Folds BatchNorm parameters into Conv weights and bias
3. Produces optimized weights for inference

This optimization:
- Reduces inference computation
- Maintains exact mathematical equivalence
- Eliminates BatchNorm overhead in production

### Examples

#### Export 10K model for production
```bash
python scripts/export_torch_model.py bytecnn_10k_20250905.pth \
    --format both \
    --name "ByteCNN-10K-UltraLight" \
    --accuracy 0.7897 \
    --output-dir ../PinyByteCNN/weights/
```

#### Export for testing
```bash
python scripts/export_torch_model.py test_model.pth \
    --format json \
    --output-dir tests/fixtures/
```

#### Export for Cloudflare Workers
```bash
python scripts/export_torch_model.py production.pth \
    --format python \
    --name "ByteCNN-Production" \
    --output-dir cloudflare/
```