#!/usr/bin/env python3
"""
Utility functions for PinyByteCNN
"""

import math


def relu_vec(x: list[float]) -> list[float]:
    """Vectorized ReLU activation function"""
    return [xi if xi > 0.0 else 0.0 for xi in x]


def text_to_fixed_bytes(text: str, max_len: int) -> list[int]:
    """Convert text to fixed-length byte array (legacy interface)"""
    return text_to_bytes(text, max_len)


def text_to_bytes(text: str, max_len: int = 512) -> list[int]:
    """
    Convert text to byte indices for model input

    Args:
        text: Input text string
        max_len: Maximum sequence length

    Returns:
        List of byte values (0-255)
    """
    # Convert to UTF-8 bytes
    byte_data = text.encode("utf-8", errors="ignore")

    # Convert to indices and truncate if needed
    indices = list(byte_data)[:max_len]

    return indices


def bytes_to_text(byte_indices: list[int]) -> str:
    """
    Convert byte indices back to text

    Args:
        byte_indices: List of byte values (0-255)

    Returns:
        Decoded text string
    """
    try:
        byte_data = bytes(byte_indices)
        return byte_data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def sliding_window_split(
    indices: list[int], window_size: int = 512, stride: int = 256
) -> list[list[int]]:
    """
    Split sequence into sliding windows

    Args:
        indices: Input sequence
        window_size: Size of each window
        stride: Stride between windows

    Returns:
        List of windows
    """
    if len(indices) <= window_size:
        return [indices]

    windows = []
    pos = 0

    while pos < len(indices):
        end_pos = min(pos + window_size, len(indices))
        window = indices[pos:end_pos]

        # Always pad windows to window_size for consistency
        if len(window) < window_size:
            window = window + [0] * (window_size - len(window))

        windows.append(window)

        # Move to next position
        pos += stride

        # If we've captured everything up to the end, break
        if pos >= len(indices):
            break

    # Ensure we have a final window if the last stride didn't cover the end
    if pos - stride + window_size < len(indices):
        final_pos = len(indices) - window_size
        if final_pos > 0 and final_pos != pos - stride:
            final_window = indices[final_pos:] + [0] * max(
                0, window_size - (len(indices) - final_pos)
            )
            windows.append(final_window)

    return windows
