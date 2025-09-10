#!/usr/bin/env python3
"""
Development environment setup for PinyByteCNN
Installs linting tools and validates setup
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n📦 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False


def main():
    """Set up development environment"""
    print("🚀 PinyByteCNN Development Setup")
    print("=" * 40)
    
    # Install development tools
    tools = [
        ("ruff", "Fast Python linter and formatter"),
        ("mypy", "Static type checker"),
        ("bandit", "Security linting"),
        ("safety", "Dependency vulnerability scanner"),
        ("coverage", "Test coverage measurement")
    ]
    
    for tool, description in tools:
        success = run_command([sys.executable, "-m", "pip", "install", tool], 
                            f"Installing {tool} - {description}")
        if not success:
            print(f"⚠️  Failed to install {tool}")
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    verification_cmds = [
        (["ruff", "--version"], "Ruff"),
        (["mypy", "--version"], "MyPy"), 
        (["bandit", "--version"], "Bandit"),
        (["safety", "--version"], "Safety"),
        (["coverage", "--version"], "Coverage")
    ]
    
    for cmd, name in verification_cmds:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            version = result.stdout.strip() or result.stderr.strip()
            print(f"✅ {name}: {version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {name}: Not found")
    
    # Test linting setup
    print("\n🧪 Testing linting setup...")
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    test_cmds = [
        (["ruff", "check", "--help"], "Ruff configuration"),
        (["mypy", "--help"], "MyPy configuration")
    ]
    
    for cmd, name in test_cmds:
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✅ {name} working")
        except subprocess.CalledProcessError:
            print(f"❌ {name} failed")
    
    # Show next steps
    print("\n🎯 Development setup complete!")
    print("\nNext steps:")
    print("  python scripts/lint.py          # Run full linting suite")  
    print("  ruff check tinybytecnn/         # Quick lint check")
    print("  ruff format .                   # Format code")
    print("  python -m unittest discover     # Run tests")
    print("  python scripts/coverage_analyzer.py  # Check test coverage")
    
    print("\n📋 Performance-focused linting rules:")
    print("  • Core library: Strict quality checks")
    print("  • Performance: Relaxed complexity limits") 
    print("  • Documentation: Optional (code density priority)")
    print("  • Tests/Scripts: Lenient rules")


if __name__ == "__main__":
    main()