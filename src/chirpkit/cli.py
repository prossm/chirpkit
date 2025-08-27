#!/usr/bin/env python3
"""
ChirpKit CLI - Command line interface for chirpkit package management and diagnostics.
"""

import sys
import platform
import subprocess
import importlib
import logging
from typing import List, Dict, Optional, Tuple
import argparse


class InstallationDiagnostics:
    """Handles installation validation and recovery for chirpkit dependencies."""
    
    def __init__(self):
        self.issues: List[Dict] = []
        self.logger = logging.getLogger(__name__)
    
    def validate_installation(self) -> List[Dict]:
        """Validate core dependencies and provide recovery suggestions."""
        self.issues = []
        
        # Check TensorFlow
        self._check_tensorflow()
        
        # Check PyTorch
        self._check_pytorch()
        
        # Check core audio libraries
        self._check_audio_libraries()
        
        # Check numpy compatibility
        self._check_numpy()
        
        return self.issues
    
    def _check_tensorflow(self):
        """Check TensorFlow installation health."""
        try:
            import tensorflow as tf
            if not hasattr(tf, '__version__'):
                self.issues.append({
                    'component': 'tensorflow',
                    'severity': 'critical',
                    'issue': 'Corrupted TensorFlow installation detected',
                    'fix': self._get_tensorflow_fix_command(),
                    'description': 'TensorFlow module exists but lacks version attribute'
                })
            else:
                # Test basic functionality
                try:
                    tf.constant([1, 2, 3])
                except Exception as e:
                    self.issues.append({
                        'component': 'tensorflow',
                        'severity': 'critical',
                        'issue': f'TensorFlow runtime error: {str(e)}',
                        'fix': self._get_tensorflow_fix_command(),
                        'description': 'TensorFlow cannot perform basic operations'
                    })
        except ImportError:
            self.issues.append({
                'component': 'tensorflow',
                'severity': 'warning',
                'issue': 'TensorFlow not installed',
                'fix': self._get_tensorflow_install_command(),
                'description': 'TensorFlow backend not available'
            })
    
    def _check_pytorch(self):
        """Check PyTorch installation health."""
        try:
            import torch
            # Test basic functionality
            try:
                torch.tensor([1, 2, 3])
            except Exception as e:
                self.issues.append({
                    'component': 'pytorch',
                    'severity': 'warning',
                    'issue': f'PyTorch runtime error: {str(e)}',
                    'fix': 'pip uninstall torch torchvision torchaudio -y && pip install torch torchvision torchaudio',
                    'description': 'PyTorch cannot perform basic operations'
                })
        except ImportError:
            self.issues.append({
                'component': 'pytorch',
                'severity': 'info',
                'issue': 'PyTorch not installed',
                'fix': 'pip install torch torchvision torchaudio',
                'description': 'PyTorch backend not available (optional)'
            })
    
    def _check_audio_libraries(self):
        """Check audio processing libraries."""
        audio_libs = ['librosa', 'soundfile']
        for lib in audio_libs:
            try:
                importlib.import_module(lib)
            except ImportError:
                self.issues.append({
                    'component': lib,
                    'severity': 'critical',
                    'issue': f'{lib} not installed',
                    'fix': f'pip install {lib}',
                    'description': f'Required for audio processing'
                })
    
    def _check_numpy(self):
        """Check numpy installation and version compatibility."""
        try:
            import numpy as np
            version = np.__version__
            major_version = int(version.split('.')[0])
            if major_version >= 2:
                self.issues.append({
                    'component': 'numpy',
                    'severity': 'warning',
                    'issue': f'NumPy {version} may have compatibility issues',
                    'fix': 'pip install "numpy>=1.21.0,<2.0.0"',
                    'description': 'NumPy 2.x may cause compatibility issues with ML libraries'
                })
        except ImportError:
            self.issues.append({
                'component': 'numpy',
                'severity': 'critical',
                'issue': 'NumPy not installed',
                'fix': 'pip install "numpy>=1.21.0,<2.0.0"',
                'description': 'Core numerical library required'
            })
    
    def _get_tensorflow_install_command(self) -> str:
        """Get platform-appropriate TensorFlow install command."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'darwin':  # macOS
            return 'pip install tensorflow-macos'
        else:
            return 'pip install tensorflow'
    
    def _get_tensorflow_fix_command(self) -> str:
        """Get TensorFlow repair command."""
        base_cmd = 'pip uninstall tensorflow tensorflow-macos keras -y && pip cache purge && '
        return base_cmd + self._get_tensorflow_install_command()
    
    def auto_fix_installation(self, component: Optional[str] = None) -> bool:
        """Attempt to fix installation issues automatically."""
        issues_to_fix = self.issues if component is None else [
            issue for issue in self.issues if issue['component'] == component
        ]
        
        success = True
        for issue in issues_to_fix:
            if issue['severity'] == 'critical':
                print(f"Attempting to fix {issue['component']}...")
                try:
                    subprocess.run(issue['fix'], shell=True, check=True)
                    print(f"✓ Fixed {issue['component']}")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to fix {issue['component']}: {e}")
                    success = False
        
        return success


class EnvironmentDetector:
    """Detects optimal installation configuration for current environment."""
    
    @staticmethod
    def detect_optimal_backend() -> Dict:
        """Detect optimal ML backend configuration for current environment."""
        system = platform.system()
        machine = platform.machine()
        python_version = sys.version_info
        
        recommendations = {
            'tensorflow_package': None,
            'tensorflow_command': None,
            'pytorch_command': 'pip install torch torchvision torchaudio',
            'notes': [],
            'platform_info': {
                'system': system,
                'machine': machine,
                'python': f"{python_version.major}.{python_version.minor}"
            }
        }
        
        if system == 'Darwin':  # macOS
            if machine in ['arm64', 'aarch64']:  # Apple Silicon
                recommendations.update({
                    'tensorflow_package': 'tensorflow-macos',
                    'tensorflow_command': 'pip install chirpkit[tensorflow-macos]',
                    'notes': [
                        'Apple Silicon detected - using tensorflow-macos',
                        'GPU acceleration available via Metal Performance Shaders',
                        'Consider installing with: pip install chirpkit[full]'
                    ]
                })
            else:  # Intel Mac
                recommendations.update({
                    'tensorflow_package': 'tensorflow-macos',
                    'tensorflow_command': 'pip install chirpkit[tensorflow-macos]',
                    'notes': [
                        'Intel macOS detected - using tensorflow-macos',
                        'Consider installing with: pip install chirpkit[full]'
                    ]
                })
        elif system == 'Linux':
            recommendations.update({
                'tensorflow_package': 'tensorflow',
                'tensorflow_command': 'pip install chirpkit[tensorflow]',
                'notes': [
                    'Linux detected - using standard tensorflow',
                    'For GPU support: pip install chirpkit[tensorflow-gpu]',
                    'Ensure CUDA drivers are installed for GPU acceleration'
                ]
            })
        elif system == 'Windows':
            recommendations.update({
                'tensorflow_package': 'tensorflow',
                'tensorflow_command': 'pip install chirpkit[tensorflow]',
                'notes': [
                    'Windows detected - using standard tensorflow',
                    'For GPU support: pip install chirpkit[tensorflow-gpu]'
                ]
            })
        
        return recommendations
    
    @staticmethod
    def print_installation_guide():
        """Print customized installation instructions for current environment."""
        rec = EnvironmentDetector.detect_optimal_backend()
        
        print("\n" + "="*60)
        print("ChirpKit Installation Recommendations")
        print("="*60)
        print(f"Platform: {rec['platform_info']['system']} {rec['platform_info']['machine']}")
        print(f"Python: {rec['platform_info']['python']}")
        print()
        
        print("Recommended Installation:")
        print(f"  {rec['tensorflow_command']}")
        print()
        
        if rec['notes']:
            print("Notes:")
            for note in rec['notes']:
                print(f"  • {note}")
            print()
        
        print("Alternative Installation Options:")
        print("  • Minimal: pip install chirpkit")
        print("  • With UI: pip install chirpkit[ui]")
        print("  • Development: pip install chirpkit[dev]")
        print("  • Complete: pip install chirpkit[full]")
        print()


def doctor():
    """Run installation diagnostics."""
    print("ChirpKit Installation Diagnostics")
    print("="*50)
    
    diagnostics = InstallationDiagnostics()
    issues = diagnostics.validate_installation()
    
    if not issues:
        print("✓ All dependencies are properly installed!")
        return 0
    
    # Group issues by severity
    critical = [i for i in issues if i['severity'] == 'critical']
    warnings = [i for i in issues if i['severity'] == 'warning']
    info = [i for i in issues if i['severity'] == 'info']
    
    if critical:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in critical:
            print(f"  ✗ {issue['component']}: {issue['issue']}")
            print(f"    Fix: {issue['fix']}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for issue in warnings:
            print(f"  ⚠ {issue['component']}: {issue['issue']}")
            print(f"    Fix: {issue['fix']}")
    
    if info:
        print("\nℹ️  INFO:")
        for issue in info:
            print(f"  ℹ {issue['component']}: {issue['issue']}")
            print(f"    Fix: {issue['fix']}")
    
    print(f"\nRun 'chirpkit-fix' to automatically fix critical issues.")
    return len(critical)


def fix():
    """Attempt to automatically fix installation issues."""
    print("ChirpKit Installation Repair")
    print("="*40)
    
    diagnostics = InstallationDiagnostics()
    issues = diagnostics.validate_installation()
    
    if not issues:
        print("✓ No issues found!")
        return 0
    
    success = diagnostics.auto_fix_installation()
    
    if success:
        print("\n✓ All critical issues have been resolved!")
        print("Run 'chirpkit-doctor' to verify the installation.")
    else:
        print("\n✗ Some issues could not be automatically resolved.")
        print("Please check the error messages above and fix manually.")
    
    return 0 if success else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ChirpKit - Insect Sound Classification Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chirpkit --help           Show this help message
  chirpkit-doctor          Check installation health
  chirpkit-fix             Auto-fix installation issues
  
For more information, visit: https://github.com/patrickmetzger/chirpkit
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='chirpkit 0.1.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Doctor command
    doctor_parser = subparsers.add_parser('doctor', help='Run installation diagnostics')
    
    # Fix command
    fix_parser = subparsers.add_parser('fix', help='Auto-fix installation issues')
    
    # Installation guide
    guide_parser = subparsers.add_parser('install-guide', help='Show platform-specific installation guide')
    
    args = parser.parse_args()
    
    if args.command == 'doctor':
        return doctor()
    elif args.command == 'fix':
        return fix()
    elif args.command == 'install-guide':
        EnvironmentDetector.print_installation_guide()
        return 0
    else:
        # Show installation guide by default
        EnvironmentDetector.print_installation_guide()
        return 0


def classify_audio_file(audio_path: str, detailed: bool = True) -> Dict:
    """
    Classify audio file - mirrors the existing CLI functionality
    
    This function uses the same logic that makes the CLI work,
    ensuring consistency between CLI and programmatic usage.
    
    Args:
        audio_path: Path to audio file to classify
        detailed: Whether to return detailed predictions
        
    Returns:
        Classification results in MoE-compatible format
    """
    try:
        # Import the existing classifier
        from .classifier import InsectClassifier
        import librosa
        import asyncio
        
        # Create classifier instance
        classifier = InsectClassifier()
        
        # Load model
        if not classifier.load_model():
            raise RuntimeError("Failed to load classifier model")
            
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Create simple audio object for compatibility
        class SimpleAudio:
            def __init__(self, waveform, sample_rate):
                self.waveform = waveform
                self.sample_rate = sample_rate
                
        processed_audio = SimpleAudio(audio, sr)
        
        # Run classification
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(classifier.classify(processed_audio, detailed=detailed))
        finally:
            loop.close()
            
        return result
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Audio classification failed: {e}")
        return {
            'model': 'ChirpKit-Error',
            'classification': {
                'is_insect': False,
                'species': 'Error',
                'confidence': 0.0,
                'family': 'unknown'
            },
            'confidence': 0.0,
            'error': str(e),
            'features': {
                'chirpkit_powered': False,
                'error': True
            }
        }


def get_classifier_instance():
    """
    Get a configured InsectClassifier instance
    
    Returns:
        Initialized InsectClassifier ready for use
    """
    from .classifier import InsectClassifier
    
    classifier = InsectClassifier()
    if not classifier.load_model():
        raise RuntimeError("Failed to initialize ChirpKit classifier")
    
    return classifier


if __name__ == '__main__':
    sys.exit(main())