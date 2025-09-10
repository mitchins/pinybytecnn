#!/usr/bin/env python3
"""
Convert PyTorch models to Optimized Multi-Layer PinyByteCNN
Using our best performing 2-layer 32KB architecture (F1=0.744)
"""

import torch
import json
import sys
import os

# Add PinyByteCNN to path
sys.path.append('/Users/mitchellcurrie/Projects/PinyByteCNN')
from tinybytecnn.multi_layer_optimized import MultiLayerByteCNN


def convert_pytorch_to_optimized_pinybytecnn(pytorch_model_path: str, 
                                           output_path: str,
                                           architecture_type: str = "2layer_32kb"):
    """
    Convert PyTorch model to optimized PinyByteCNN format
    
    Args:
        pytorch_model_path: Path to .pth file
        output_path: Path for output .json file
        architecture_type: "2layer_32kb" or "3layer_32kb"
    """
    
    print(f"🔄 Converting {pytorch_model_path} to Optimized PinyByteCNN...")
    print(f"   Architecture: {architecture_type}")
    
    # Load PyTorch weights
    try:
        state_dict = torch.load(pytorch_model_path, map_location='cpu')
        print(f"✅ Loaded PyTorch model")
    except Exception as e:
        print(f"❌ Failed to load PyTorch model: {e}")
        return False
    
    # Create optimized model
    if architecture_type == "2layer_32kb":
        model = MultiLayerByteCNN.create_2layer_32kb(max_len=512)
        expected_layers = 2
    elif architecture_type == "3layer_32kb":
        model = MultiLayerByteCNN.create_3layer_32kb(max_len=512) 
        expected_layers = 3
    else:
        raise ValueError(f"Unknown architecture: {architecture_type}")
    
    print(f"✅ Created optimized model with {len(model.layers_config)} layers")
    
    # Convert weights to the format expected by optimized model
    converted_weights = {}
    
    # Extract embedding weights
    if hasattr(state_dict, 'embedding.weight') or 'embedding.weight' in state_dict:
        embedding_key = 'embedding.weight' if 'embedding.weight' in state_dict else 'embedding.weight'
        embedding_weights = state_dict[embedding_key].detach().numpy().tolist()
        converted_weights["embedding"] = embedding_weights
        print(f"   ✅ Converted embedding: {len(embedding_weights)}x{len(embedding_weights[0])}")
    
    # Extract conv layer weights with batch normalization (handling different naming conventions)
    conv_layer_mapping = [
        ("conv1d_1.weight", "conv1d_1.bias", "bn1", "conv1_weight", "conv1_bias", "bn1"),
        ("conv1d_2.weight", "conv1d_2.bias", "bn2", "conv2_weight", "conv2_bias", "bn2"),
        ("conv1d_3.weight", "conv1d_3.bias", "bn3", "conv3_weight", "conv3_bias", "bn3")
    ]
    
    for i, (weight_key, bias_key, bn_key, out_weight_key, out_bias_key, out_bn_key) in enumerate(conv_layer_mapping):
        if i >= expected_layers:
            break
            
        if weight_key in state_dict and bias_key in state_dict:
            # Convert from PyTorch format [out_ch, in_ch, kernel] to our format
            conv_weight = state_dict[weight_key].detach().numpy().tolist()
            conv_bias = state_dict[bias_key].detach().numpy().tolist()
            
            converted_weights[out_weight_key] = conv_weight
            converted_weights[out_bias_key] = conv_bias
            
            # Extract batch normalization parameters if available
            bn_weight_key = f"{bn_key}.weight"
            bn_bias_key = f"{bn_key}.bias"
            bn_mean_key = f"{bn_key}.running_mean"
            bn_var_key = f"{bn_key}.running_var"
            
            if all(k in state_dict for k in [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]):
                converted_weights[f"{out_bn_key}_weight"] = state_dict[bn_weight_key].detach().numpy().tolist()
                converted_weights[f"{out_bn_key}_bias"] = state_dict[bn_bias_key].detach().numpy().tolist()
                converted_weights[f"{out_bn_key}_running_mean"] = state_dict[bn_mean_key].detach().numpy().tolist()
                converted_weights[f"{out_bn_key}_running_var"] = state_dict[bn_var_key].detach().numpy().tolist()
                print(f"   ✅ Converted BN layer {i+1}")
            
            shape_info = f"{len(conv_weight)}x{len(conv_weight[0])}x{len(conv_weight[0][0])}"
            print(f"   ✅ Converted conv layer {i+1}: {shape_info}")
    
    # Extract classifier weights
    classifier_keys = [
        ("classifier.weight", "classifier.bias", "classifier_weight", "classifier_bias"),
        ("fc1.weight", "fc1.bias", "classifier_weight", "classifier_bias"),  # Alternative naming
    ]
    
    for weight_key, bias_key, out_weight_key, out_bias_key in classifier_keys:
        if weight_key in state_dict and bias_key in state_dict:
            classifier_weight = state_dict[weight_key].detach().numpy().tolist()
            classifier_bias = state_dict[bias_key].detach().numpy().tolist()
            
            converted_weights[out_weight_key] = classifier_weight
            converted_weights[out_bias_key] = classifier_bias
            
            shape_info = f"{len(classifier_weight)}x{len(classifier_weight[0])}"
            print(f"   ✅ Converted classifier: {shape_info}")
            break
    
    # Extract output weights
    output_keys = [
        ("output.weight", "output.bias", "output_weight", "output_bias"),
        ("fc2.weight", "fc2.bias", "output_weight", "output_bias"),  # Alternative naming
    ]
    
    for weight_key, bias_key, out_weight_key, out_bias_key in output_keys:
        if weight_key in state_dict and bias_key in state_dict:
            output_weight = state_dict[weight_key].detach().numpy().tolist()
            output_bias = state_dict[bias_key].detach().numpy().tolist()
            
            converted_weights[out_weight_key] = output_weight
            converted_weights[out_bias_key] = output_bias
            
            shape_info = f"{len(output_weight)}x{len(output_weight[0])}"
            print(f"   ✅ Converted output: {shape_info}")
            break
    
    # Add model metadata
    converted_weights["model_info"] = {
        "architecture_type": architecture_type,
        "layers": expected_layers,
        "max_seq_len": 512,
        "vocab_size": 256,
        "source_model": pytorch_model_path,
        "conversion_format": "optimized_multi_layer_v1"
    }
    
    # Save to JSON
    try:
        with open(output_path, 'w') as f:
            json.dump(converted_weights, f, indent=2)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"💾 Saved to: {output_path} ({file_size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        return False
    
    # Test loading with optimized model
    try:
        model.load_weights_from_dict(converted_weights)
        
        # Quick test
        test_result = model.predict("This is a test message", strategy="truncate")
        print(f"🧪 Test prediction: {test_result:.4f}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Conversion succeeded but loading test failed: {e}")
        return False


def convert_best_models():
    """Convert our best performing models to optimized format"""
    
    print("🚀 Converting Best Models to Optimized PinyByteCNN")
    print("=" * 60)
    
    # Models to convert (from our sweeps)
    models_to_convert = [
        # From breakthrough sweep - our current best
        {
            "source": "32kb_2L_breakthrough_f1_0.744.pth",
            "target": "/Users/mitchellcurrie/Projects/PinyByteCNN/weights/optimized_32kb_2L_f1_744.json",
            "arch": "2layer_32kb",
            "description": "Best 2-layer model (F1=0.744)"
        },
        
        # From original sliding window (if available)
        {
            "source": "bytecnn_32kb_sliding_pytorch_f1_0.791.pth", 
            "target": "/Users/mitchellcurrie/Projects/PinyByteCNN/weights/optimized_32kb_3L_f1_791.json",
            "arch": "3layer_32kb",
            "description": "Original breakthrough model (F1=0.791)"
        }
    ]
    
    results = {}
    
    for model_config in models_to_convert:
        source_path = model_config["source"]
        
        # Check if source exists
        if not os.path.exists(source_path):
            print(f"⚠️  Skipping {source_path} - file not found")
            continue
            
        print(f"\n📦 Converting: {model_config['description']}")
        
        success = convert_pytorch_to_optimized_pinybytecnn(
            source_path,
            model_config["target"], 
            model_config["arch"]
        )
        
        results[model_config["description"]] = {
            "success": success,
            "source": source_path,
            "target": model_config["target"],
            "architecture": model_config["arch"]
        }
    
    # Summary
    print(f"\n🏆 CONVERSION SUMMARY")
    print("=" * 40)
    
    successful = [desc for desc, info in results.items() if info["success"]]
    failed = [desc for desc, info in results.items() if not info["success"]]
    
    print(f"✅ Successful: {len(successful)}")
    for desc in successful:
        print(f"   • {desc}")
        print(f"     → {results[desc]['target']}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for desc in failed:
            print(f"   • {desc}")
    
    print(f"\n🎯 RECOMMENDED USAGE:")
    if successful:
        best_model = successful[0]  # First successful is typically best
        target_path = results[best_model]['target']
        print(f"   Load: {target_path}")
        print(f"   Architecture: {results[best_model]['architecture']}")
        print(f"   Strategies: truncate (fast), attention (high accuracy)")
    
    return results


def benchmark_optimized_model():
    """Benchmark the optimized model performance"""
    
    print("\n🏃 Benchmarking Optimized Multi-Layer PinyByteCNN")
    print("=" * 50)
    
    # Create optimized model
    model = MultiLayerByteCNN.create_2layer_32kb(max_len=512)
    
    # Try to load weights if available
    weights_paths = [
        "/Users/mitchellcurrie/Projects/PinyByteCNN/weights/optimized_32kb_2L_f1_744.json",
        "weights_32kb_sliding_f1_0.744.json"  # Alternative location
    ]
    
    weights_loaded = False
    for weights_path in weights_paths:
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    weights = json.load(f)
                model.load_weights_from_dict(weights)
                weights_loaded = True
                print(f"✅ Loaded weights from: {weights_path}")
                break
            except Exception as e:
                print(f"⚠️  Failed to load {weights_path}: {e}")
    
    if not weights_loaded:
        print("⚠️  No weights loaded - using random initialization for benchmarking")
    
    # Run benchmark
    from tinybytecnn.multi_layer_optimized import OptimizedArchitectures
    
    print("🔄 Running inference benchmark...")
    benchmark_results = OptimizedArchitectures.benchmark_inference_speed(model, num_samples=1000)
    
    print(f"📊 PERFORMANCE RESULTS:")
    print("-" * 30)
    for metric, value in benchmark_results.items():
        if "rps" in metric:
            print(f"   {metric.replace('_', ' ').title()}: {value:.1f}")
        else:
            print(f"   {metric.replace('_', ' ').title()}: {value:.2f}")
    
    # Test different strategies
    test_texts = [
        "Short toxic message",
        "This is a much longer message that should trigger sliding window processing " * 10,
        "Normal message with no toxicity detected here"
    ]
    
    print(f"\n🧪 STRATEGY COMPARISON:")
    print("-" * 40)
    
    for i, text in enumerate(test_texts):
        text_desc = f"Text {i+1} ({len(text)} chars)"
        print(f"{text_desc}:")
        
        for strategy in ["truncate", "average", "attention"]:
            try:
                prob = model.predict(text, strategy=strategy)
                print(f"   {strategy:>9}: {prob:.4f}")
            except Exception as e:
                print(f"   {strategy:>9}: Error - {e}")
    
    return benchmark_results


if __name__ == "__main__":
    # Convert models
    conversion_results = convert_best_models()
    
    # Benchmark performance
    benchmark_results = benchmark_optimized_model()