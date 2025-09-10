#!/usr/bin/env python3
"""
Universal PyTorch to PinyByteCNN Export Script
Supports both JSON and Static Python export formats
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os
import sys


def fold_batchnorm_into_conv(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm parameters into Conv layer for inference optimization"""
    std = torch.sqrt(bn_var + eps)
    norm_factor = bn_weight / std
    
    folded_weight = conv_weight * norm_factor.view(-1, 1, 1)
    
    if conv_bias is None:
        conv_bias = torch.zeros(conv_weight.size(0))
    folded_bias = (conv_bias - bn_mean) * norm_factor + bn_bias
    
    return folded_weight, folded_bias


def export_to_json(state_dict, output_path, model_info=None):
    """Export PyTorch model weights to JSON format"""
    print(f"Exporting to JSON: {output_path}")
    
    # Prepare weights dictionary
    weights = {}
    
    # Handle different model architectures
    if 'embedding.weight' in state_dict:
        weights['embedding'] = state_dict['embedding.weight'].detach().cpu().numpy().tolist()
    
    # Check for conv layers with potential BatchNorm
    if 'conv1.weight' in state_dict:
        if 'bn1.weight' in state_dict:
            # Fold BatchNorm into Conv
            folded_weight, folded_bias = fold_batchnorm_into_conv(
                conv_weight=state_dict['conv1.weight'],
                conv_bias=state_dict.get('conv1.bias'),
                bn_weight=state_dict['bn1.weight'],
                bn_bias=state_dict['bn1.bias'],
                bn_mean=state_dict['bn1.running_mean'],
                bn_var=state_dict['bn1.running_var']
            )
            weights['conv1_weight'] = folded_weight.detach().cpu().numpy().tolist()
            weights['conv1_bias'] = folded_bias.detach().cpu().numpy().tolist()
        else:
            weights['conv1_weight'] = state_dict['conv1.weight'].detach().cpu().numpy().tolist()
            if 'conv1.bias' in state_dict:
                weights['conv1_bias'] = state_dict['conv1.bias'].detach().cpu().numpy().tolist()
    
    # Handle classifier/dense layers
    if 'classifier.weight' in state_dict:
        weights['classifier_weight'] = state_dict['classifier.weight'].detach().cpu().numpy().tolist()
        weights['classifier_bias'] = state_dict['classifier.bias'].detach().cpu().numpy().tolist()
    
    if 'fc1.weight' in state_dict:
        weights['fc1_weight'] = state_dict['fc1.weight'].detach().cpu().numpy().tolist()
        weights['fc1_bias'] = state_dict['fc1.bias'].detach().cpu().numpy().tolist()
    
    if 'fc2.weight' in state_dict:
        weights['fc2_weight'] = state_dict['fc2.weight'].detach().cpu().numpy().tolist()
        weights['fc2_bias'] = state_dict['fc2.bias'].detach().cpu().numpy().tolist()
    
    # Handle output layer
    if 'output.weight' in state_dict:
        weights['output_weight'] = state_dict['output.weight'].detach().cpu().numpy().tolist()
        weights['output_bias'] = state_dict['output.bias'].detach().cpu().numpy().tolist()
    
    # Create full JSON structure
    json_data = {
        "model_info": model_info or {
            "name": "ByteCNN-Export",
            "version": "1.0.0",
            "exported": datetime.now().isoformat(),
            "parameters": sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
        },
        "weights": weights,
        "architecture": detect_architecture(state_dict)
    }
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"Saved JSON: {output_path} ({file_size:.1f} KB)")
    print(f"Total parameters: {json_data['model_info']['parameters']:,}")


def export_to_static_python(state_dict, output_path, model_info=None):
    """Export PyTorch model weights to static Python format"""
    print(f"Exporting to Static Python: {output_path}")
    
    # Start building Python code
    code = f'''#!/usr/bin/env python3
"""
ByteCNN Static Weights
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Pure Python implementation for edge deployment
"""

# Model configuration
MODEL_INFO = {{
    "name": "{model_info.get('name', 'ByteCNN') if model_info else 'ByteCNN'}",
    "version": "{model_info.get('version', '1.0.0') if model_info else '1.0.0'}",
    "parameters": {sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))},
    "exported": "{datetime.now().isoformat()}"
}}

'''
    
    # Export embedding weights
    if 'embedding.weight' in state_dict:
        code += "# Embedding weights [vocab_size, embed_dim]\n"
        code += "EMBEDDING_WEIGHT = [\n"
        for row in state_dict['embedding.weight'].detach().cpu().numpy().tolist():
            code += f"    {row},\n"
        code += "]\n\n"
    
    # Export conv weights (with BatchNorm folding if needed)
    if 'conv1.weight' in state_dict:
        if 'bn1.weight' in state_dict:
            folded_weight, folded_bias = fold_batchnorm_into_conv(
                conv_weight=state_dict['conv1.weight'],
                conv_bias=state_dict.get('conv1.bias'),
                bn_weight=state_dict['bn1.weight'],
                bn_bias=state_dict['bn1.bias'],
                bn_mean=state_dict['bn1.running_mean'],
                bn_var=state_dict['bn1.running_var']
            )
            code += "# Conv1 weights [out_channels, in_channels, kernel_size] - BatchNorm folded\n"
            code += "CONV1_WEIGHT = [\n"
            for filter_weights in folded_weight.detach().cpu().numpy().tolist():
                code += f"    {filter_weights},\n"
            code += "]\n\n"
            code += "# Conv1 bias [out_channels] - BatchNorm folded\n"
            code += f"CONV1_BIAS = {folded_bias.detach().cpu().numpy().tolist()}\n\n"
        else:
            code += "# Conv1 weights [out_channels, in_channels, kernel_size]\n"
            code += "CONV1_WEIGHT = [\n"
            for filter_weights in state_dict['conv1.weight'].detach().cpu().numpy().tolist():
                code += f"    {filter_weights},\n"
            code += "]\n\n"
            if 'conv1.bias' in state_dict:
                code += "# Conv1 bias [out_channels]\n"
                code += f"CONV1_BIAS = {state_dict['conv1.bias'].detach().cpu().numpy().tolist()}\n\n"
    
    # Export classifier/dense layers
    if 'classifier.weight' in state_dict:
        code += "# Classifier weights [out_features, in_features]\n"
        code += "CLASSIFIER_WEIGHT = [\n"
        for row in state_dict['classifier.weight'].detach().cpu().numpy().tolist():
            code += f"    {row},\n"
        code += "]\n\n"
        code += "# Classifier bias [out_features]\n"
        code += f"CLASSIFIER_BIAS = {state_dict['classifier.bias'].detach().cpu().numpy().tolist()}\n\n"
    
    if 'fc1.weight' in state_dict:
        code += "# FC1 weights [out_features, in_features]\n"
        code += "FC1_WEIGHT = [\n"
        for row in state_dict['fc1.weight'].detach().cpu().numpy().tolist():
            code += f"    {row},\n"
        code += "]\n\n"
        code += f"FC1_BIAS = {state_dict['fc1.bias'].detach().cpu().numpy().tolist()}\n\n"
    
    if 'fc2.weight' in state_dict:
        code += "# FC2 weights [out_features, in_features]\n"
        code += "FC2_WEIGHT = [\n"
        for row in state_dict['fc2.weight'].detach().cpu().numpy().tolist():
            code += f"    {row},\n"
        code += "]\n\n"
        code += f"FC2_BIAS = {state_dict['fc2.bias'].detach().cpu().numpy().tolist()}\n\n"
    
    # Export output layer
    if 'output.weight' in state_dict:
        code += "# Output weights [out_features, in_features]\n"
        code += f"OUTPUT_WEIGHT = {state_dict['output.weight'].detach().cpu().numpy().tolist()}\n\n"
        code += "# Output bias [out_features]\n"
        code += f"OUTPUT_BIAS = {state_dict['output.bias'].detach().cpu().numpy().tolist()}\n"
    
    # Save Python file
    with open(output_path, 'w') as f:
        f.write(code)
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"Saved Static Python: {output_path} ({file_size:.1f} KB)")
    total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
    print(f"Total parameters: {total_params:,}")


def detect_architecture(state_dict):
    """Detect model architecture from state dict"""
    arch = {}
    
    if 'embedding.weight' in state_dict:
        shape = state_dict['embedding.weight'].shape
        arch['vocab_size'] = shape[0]
        arch['embed_dim'] = shape[1]
    
    if 'conv1.weight' in state_dict:
        shape = state_dict['conv1.weight'].shape
        arch['conv_filters'] = shape[0]
        arch['kernel_size'] = shape[2]
    
    if 'classifier.weight' in state_dict:
        shape = state_dict['classifier.weight'].shape
        arch['dense_dim'] = shape[0]
    
    if 'fc1.weight' in state_dict:
        shape = state_dict['fc1.weight'].shape
        arch['hidden_dim'] = shape[0]
    
    arch['max_len'] = 512  # Default assumption
    
    return arch


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to PinyByteCNN formats')
    parser.add_argument('model_path', help='Path to PyTorch .pth file')
    parser.add_argument('--format', choices=['json', 'python', 'both'], default='both',
                       help='Export format (default: both)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--name', help='Model name for metadata')
    parser.add_argument('--version', default='1.0.0', help='Model version')
    parser.add_argument('--accuracy', type=float, help='Model accuracy for metadata')
    
    args = parser.parse_args()
    
    # Load PyTorch model
    print(f"Loading model from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location='cpu')
    
    # Prepare model info
    model_info = {
        'name': args.name or os.path.basename(args.model_path).replace('.pth', ''),
        'version': args.version,
        'parameters': sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
    }
    
    if args.accuracy:
        model_info['accuracy'] = args.accuracy
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export based on format
    base_name = os.path.basename(args.model_path).replace('.pth', '')
    
    if args.format in ['json', 'both']:
        json_path = os.path.join(args.output_dir, f"{base_name}.json")
        export_to_json(state_dict, json_path, model_info)
    
    if args.format in ['python', 'both']:
        python_path = os.path.join(args.output_dir, f"{base_name}_weights.py")
        export_to_static_python(state_dict, python_path, model_info)
    
    print("\nExport complete!")
    
    # Test imports if Python format was exported
    if args.format in ['python', 'both']:
        print("\nTesting Python import...")
        sys.path.insert(0, args.output_dir)
        module_name = f"{base_name}_weights"
        try:
            exec(f"import {module_name}")
            print(f"Successfully imported {module_name}")
            exec(f"print(f'Model info: {{{module_name}.MODEL_INFO}}}')")
        except Exception as e:
            print(f"Warning: Could not import {module_name}: {e}")


if __name__ == "__main__":
    main()