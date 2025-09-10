# Deployment Guide

## Overview

PinyByteCNN is designed for deployment in resource-constrained environments. This guide covers deployment strategies for various platforms.

## Cloudflare Workers

### Setup

1. Create a new Cloudflare Worker
2. Copy the PinyByteCNN source files
3. Import the model weights
4. Configure the request handler

### Example Worker

```javascript
import { EmbeddedByteCNN } from './embedded_bytecnn_10k.js';

export default {
  async fetch(request, env, ctx) {
    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 });
    }

    const { text } = await request.json();
    const model = new EmbeddedByteCNN();
    const score = model.predict(text);

    return Response.json({
      toxic: score > 0.5,
      confidence: score,
      model: "ByteCNN-10K"
    });
  }
};
```

### Performance Considerations

- Use ByteCNN-10K for sub-10ms inference
- Enable gzip compression for weight files
- Implement request batching for efficiency
- Monitor CPU usage limits (10ms default)

## AWS Lambda

### Deployment Package

```bash
# Create deployment package
mkdir lambda-package
cp -r tinybytecnn/ lambda-package/
cp lambda_handler.py lambda-package/
cd lambda-package && zip -r ../bytecnn-lambda.zip .
```

### Handler Example

```python
import json
from tinybytecnn.model import ByteCNN

# Initialize model once (outside handler)
model = ByteCNN(vocab_size=256, embed_dim=14, conv_filters=28, 
                conv_kernel_size=3, hidden_dim=48)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        text = body.get('text', '')
        
        score = model.predict(text)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'toxic': score > 0.5,
                'confidence': float(score)
            })
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
```

### Configuration

- Runtime: Python 3.9+
- Memory: 128MB (sufficient for ByteCNN-10K)
- Timeout: 15 seconds (typically needs <1s)
- Environment variables: MODEL_PATH for weight loading

## Google Cloud Functions

### Requirements File

```
# requirements.txt - empty for pure Python deployment
```

### Function Code

```python
import functions_framework
from tinybytecnn.multi_layer_optimized import MultiLayerByteCNN

# Initialize model globally
model = None

def get_model():
    global model
    if model is None:
        layers = [{"in_channels": 14, "out_channels": 28, "kernel_size": 3}]
        model = MultiLayerByteCNN(layers_config=layers, hidden_dim=48)
    return model

@functions_framework.http
def classify_text(request):
    if request.method != 'POST':
        return {'error': 'Method not allowed'}, 405
    
    data = request.get_json()
    text = data.get('text', '')
    
    model = get_model()
    score = model.predict(text)
    
    return {
        'toxic': score > 0.5,
        'confidence': float(score),
        'model': 'PinyByteCNN'
    }
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy source code
COPY tinybytecnn/ ./tinybytecnn/
COPY app.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

### Application Server

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from tinybytecnn.model import ByteCNN

class ClassifyHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.model = ByteCNN(vocab_size=256, embed_dim=14, conv_filters=28,
                           conv_kernel_size=3, hidden_dim=48)
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        text = data.get('text', '')
        score = self.model.predict(text)
        
        response = {
            'toxic': score > 0.5,
            'confidence': float(score)
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), ClassifyHandler)
    server.serve_forever()
```

## Mobile/IoT Deployment

### Android (via Chaquopy)

```python
# android_bytecnn.py
from tinybytecnn.model import ByteCNN

class AndroidByteCNN:
    def __init__(self):
        self.model = ByteCNN(vocab_size=256, embed_dim=14, conv_filters=28,
                           conv_kernel_size=3, hidden_dim=48)
    
    def classify(self, text):
        score = self.model.predict(text)
        return {
            'is_toxic': score > 0.5,
            'confidence': score
        }
```

### iOS (via Kivy/BeeWare)

```python
# ios_wrapper.py
import json
from tinybytecnn.multi_layer_optimized import MultiLayerByteCNN

def create_model():
    layers = [{"in_channels": 14, "out_channels": 28, "kernel_size": 3}]
    return MultiLayerByteCNN(layers_config=layers, hidden_dim=48)

def classify_text_ios(text_input):
    model = create_model()
    score = model.predict(text_input)
    
    return json.dumps({
        'toxic': score > 0.5,
        'confidence': float(score)
    })
```

## Performance Optimization

### Model Selection

- **Ultra-low latency**: ByteCNN-10K (10ms)
- **Balanced**: ByteCNN-15K (15ms)
- **High accuracy**: ByteCNN-32K (25ms)

### Memory Optimization

```python
# Minimize memory allocations
class OptimizedByteCNN(ByteCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-allocate buffers
        self._temp_buffer = [0.0] * self.max_len
        self._embed_buffer = [[0.0] * self.embed_dim for _ in range(self.max_len)]
```

### Caching Strategies

```python
from functools import lru_cache

class CachedByteCNN:
    def __init__(self):
        self.model = ByteCNN(...)
    
    @lru_cache(maxsize=1000)
    def predict_cached(self, text):
        return self.model.predict(text)
```

## Monitoring and Observability

### Logging

```python
import logging
import time

class InstrumentedByteCNN:
    def __init__(self):
        self.model = ByteCNN(...)
        self.logger = logging.getLogger('bytecnn')
    
    def predict(self, text):
        start_time = time.time()
        score = self.model.predict(text)
        inference_time = time.time() - start_time
        
        self.logger.info(f"Prediction: {score:.4f}, Time: {inference_time*1000:.2f}ms")
        return score
```

### Health Checks

```python
def health_check():
    try:
        model = ByteCNN(vocab_size=256, embed_dim=14, conv_filters=28,
                       conv_kernel_size=3, hidden_dim=48)
        
        # Test prediction
        test_score = model.predict("health check")
        
        return {
            'status': 'healthy',
            'model_loaded': True,
            'test_score': float(test_score)
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }
```

## Security Considerations

### Input Validation

```python
def validate_input(text):
    if not isinstance(text, str):
        raise ValueError("Input must be string")
    
    if len(text.encode('utf-8')) > 10000:  # 10KB limit
        raise ValueError("Input too large")
    
    return text
```

### Rate Limiting

```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=100, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def allow_request(self, client_id):
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests
        client_requests[:] = [req for req in client_requests if now - req < self.window]
        
        if len(client_requests) >= self.max_requests:
            return False
        
        client_requests.append(now)
        return True
```