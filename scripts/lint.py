#!/usr/bin/env python3
"""
Local linting script for PinyByteCNN development
Runs the same checks as CI with performance-focused configurations
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, allow_failure=False):
    """Run a command and report results"""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0 and not allow_failure:
            print(f"❌ {description} failed with code {result.returncode}")
            return False
        elif result.returncode != 0:
            print(f"⚠️  {description} had issues (allowed)")
        else:
            print(f"✅ {description} passed")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def main():
    """Main linting routine"""
    print("🚀 PinyByteCNN Linting - Performance Optimized")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    success = True
    
    # Core library - strict linting
    success &= run_command(
        ["ruff", "check", "tinybytecnn/"],
        "Ruff lint check (tinybytecnn - strict)"
    )
    
    # Other files - lenient linting
    success &= run_command(
        ["ruff", "check", "tests/", "benchmarks/", "scripts/"],
        "Ruff lint check (tests/benchmarks/scripts - lenient)",
        allow_failure=True
    )
    
    # Format check
    success &= run_command(
        ["ruff", "format", "--check", "--diff", "."],
        "Ruff format check"
    )
    
    # Type checking (lenient)
    success &= run_command(
        ["mypy", "tinybytecnn/"],
        "MyPy type check",
        allow_failure=True
    )
    
    # Security check
    success &= run_command(
        ["bandit", "-r", "tinybytecnn/"],
        "Bandit security check",
        allow_failure=True
    )
    
    print("\n" + "=" * 60)
    if success:
        print("🎯 All critical checks passed!")
        print("📊 Performance-focused linting complete")
    else:
        print("⚠️  Some checks failed - review output above")
        print("💡 Core library (tinybytecnn/) must pass all checks")
        
    print("\n📖 Key principles:")
    print("  • Core library: Strict linting for reliability")  
    print("  • Performance: Complexity rules relaxed for optimization")
    print("  • Tests/Scripts: Lenient rules for development tools")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())