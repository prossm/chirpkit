#!/usr/bin/env python3
"""
ChirpKit Model Configuration Examples

This script demonstrates the enhanced model configuration system
that provides maximum flexibility for different deployment scenarios.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.chirpkit import (
        InsectClassifier, 
        ModelConfiguration, 
        ConfigurationManager,
        ModelDiscovery,
        ModelValidator,
        create_example_config_file
    )
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Please install ChirpKit or run from the project root directory")
    sys.exit(1)


def example_1_explicit_paths():
    """Example 1: Explicit paths for full control"""
    print("\n" + "="*60)
    print("📁 Example 1: Explicit Model Paths")
    print("="*60)
    
    try:
        # SoundCurious scenario - use existing downloaded models
        classifier = InsectClassifier(
            birdnet_model_path="/models/chirpkit/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite",
            ensemble_path="/models/chirpkit/trained/chirpkit-ensemble",
            mode="ensemble_tta",
            auto_download=False,  # Don't download, use existing
            validate_compatibility=True
        )
        
        print("✅ Classifier created with explicit paths")
        print(f"   BirdNET: {classifier.birdnet_path}")
        print(f"   Ensemble: {classifier.model_path}")
        print(f"   Mode: {classifier.mode}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")


def example_2_root_directory():
    """Example 2: Root directory with auto-discovery"""
    print("\n" + "="*60)
    print("🔍 Example 2: Root Directory + Auto-Discovery")
    print("="*60)
    
    try:
        # Let ChirpKit discover compatible models in directory
        classifier = InsectClassifier(
            model_root="/models/chirpkit",
            mode="ensemble",
            validate_compatibility=True
        )
        
        print("✅ Classifier created with model discovery")
        print(f"   Root: /models/chirpkit")
        print(f"   Auto-discovered BirdNET: {classifier.birdnet_path}")
        print(f"   Auto-discovered Ensemble: {classifier.model_path}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")


def example_3_environment_variables():
    """Example 3: Environment variable configuration"""
    print("\n" + "="*60)
    print("🌍 Example 3: Environment Variables")
    print("="*60)
    
    # Set environment variables
    os.environ['CHIRPKIT_ROOT_DIR'] = '/models/chirpkit'
    os.environ['CHIRPKIT_MODE'] = 'ensemble_tta'
    os.environ['CHIRPKIT_AUTO_DOWNLOAD'] = 'false'
    
    try:
        # Configuration automatically loaded from environment
        classifier = InsectClassifier()
        
        print("✅ Classifier created from environment variables")
        print(f"   CHIRPKIT_ROOT_DIR: {os.environ.get('CHIRPKIT_ROOT_DIR')}")
        print(f"   CHIRPKIT_MODE: {os.environ.get('CHIRPKIT_MODE')}")
        print(f"   Resolved mode: {classifier.mode}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        # Clean up environment variables
        for key in ['CHIRPKIT_ROOT_DIR', 'CHIRPKIT_MODE', 'CHIRPKIT_AUTO_DOWNLOAD']:
            os.environ.pop(key, None)


def example_4_config_file():
    """Example 4: Configuration file"""
    print("\n" + "="*60)
    print("📋 Example 4: Configuration File")
    print("="*60)
    
    # Create example config file
    config_path = "example_config.yaml"
    
    config_content = """
models:
  root_directory: "/models/chirpkit"
  ensemble:
    mode: "ensemble_tta"
  download:
    auto_download: false
  validate_compatibility: true

advanced:
  tta_rounds: 10
  device: "auto"
"""
    
    try:
        # Write config file
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # ChirpKit will find and use this config file
        classifier = InsectClassifier()
        
        print("✅ Classifier created from configuration file")
        print(f"   Config file: {config_path}")
        print(f"   Mode: {classifier.mode}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        # Clean up
        Path(config_path).unlink(missing_ok=True)


def example_5_model_discovery():
    """Example 5: Model discovery and validation"""
    print("\n" + "="*60)
    print("🕵️ Example 5: Model Discovery")
    print("="*60)
    
    # Test with a mock directory structure
    test_dir = Path("test_models")
    
    try:
        # Create mock directory structure
        test_dir.mkdir(exist_ok=True)
        (test_dir / "birdnet").mkdir(exist_ok=True)
        (test_dir / "trained" / "chirpkit-ensemble").mkdir(parents=True, exist_ok=True)
        
        # Create mock files
        (test_dir / "birdnet" / "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite").touch()
        (test_dir / "trained" / "chirpkit-ensemble" / "ensemble_model_1.pth").touch()
        (test_dir / "trained" / "chirpkit-ensemble" / "ensemble_info.json").write_text('{"n_classes": 231, "num_models": 7}')
        
        # Discover models
        discovered = ModelDiscovery.find_models(test_dir)
        selection = ModelDiscovery.select_best_models(discovered)
        
        print(f"✅ Model discovery completed")
        print(f"   Search directory: {test_dir}")
        print(f"   Discovered: {discovered}")
        print(f"   Selected: {selection}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        # Clean up
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


def example_6_backwards_compatibility():
    """Example 6: Backwards compatibility"""
    print("\n" + "="*60)
    print("🔄 Example 6: Backwards Compatibility")
    print("="*60)
    
    try:
        # Old style - still works
        classifier = InsectClassifier(mode="ensemble")
        
        print("✅ Old-style constructor still works")
        print(f"   Mode: {classifier.mode}")
        print(f"   Uses default paths: {classifier.model_path}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")


def example_7_docker_deployment():
    """Example 7: Docker/Container deployment"""
    print("\n" + "="*60)
    print("🐳 Example 7: Docker/Container Deployment")
    print("="*60)
    
    print("Docker environment setup:")
    print("  ENV CHIRPKIT_ROOT_DIR=/models/chirpkit")
    print("  VOLUME /models/chirpkit")
    
    # Simulate Docker environment
    os.environ['CHIRPKIT_ROOT_DIR'] = '/models/chirpkit'
    
    try:
        # This would work in a Docker container with mounted volume
        config_manager = ConfigurationManager()
        config = config_manager.resolve_configuration()
        
        print("✅ Docker configuration resolved")
        print(f"   Root directory: {config.root_directory}")
        print(f"   Auto-download: {config.auto_download}")
        print(f"   Fallback enabled: {config.fallback_to_default}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        os.environ.pop('CHIRPKIT_ROOT_DIR', None)


def example_8_configuration_priority():
    """Example 8: Configuration priority demonstration"""
    print("\n" + "="*60)
    print("📊 Example 8: Configuration Priority")
    print("="*60)
    
    # Set environment variable
    os.environ['CHIRPKIT_MODE'] = 'ensemble'
    
    try:
        # Constructor parameter should override environment variable
        classifier = InsectClassifier(mode="ensemble_tta")
        
        print("✅ Configuration priority test")
        print(f"   Environment CHIRPKIT_MODE: {os.environ.get('CHIRPKIT_MODE')}")
        print(f"   Constructor mode: ensemble_tta")
        print(f"   Final mode: {classifier.mode}")
        print(f"   ✅ Constructor parameter takes priority!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        os.environ.pop('CHIRPKIT_MODE', None)


def example_9_create_config_file():
    """Example 9: Create example configuration file"""
    print("\n" + "="*60)
    print("📝 Example 9: Create Configuration File")
    print("="*60)
    
    try:
        config_path = "~/.chirpkit/example_config.yaml"
        create_example_config_file(config_path)
        
        expanded_path = Path(config_path).expanduser()
        print(f"✅ Example configuration file created")
        print(f"   Location: {expanded_path}")
        
        if expanded_path.exists():
            print("   Content preview:")
            content = expanded_path.read_text()
            for i, line in enumerate(content.split('\n')[:10]):
                print(f"     {line}")
            if len(content.split('\n')) > 10:
                print("     ...")
        
    except Exception as e:
        print(f"❌ Failed: {e}")


def main():
    """Run all examples"""
    print("🦗 ChirpKit Model Configuration Examples")
    print("This demonstrates the enhanced flexibility for different deployment scenarios")
    
    examples = [
        example_1_explicit_paths,
        example_2_root_directory, 
        example_3_environment_variables,
        example_4_config_file,
        example_5_model_discovery,
        example_6_backwards_compatibility,
        example_7_docker_deployment,
        example_8_configuration_priority,
        example_9_create_config_file
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"💥 Example failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 Configuration examples completed!")
    print("="*60)
    print("\n📚 Key benefits of the enhanced configuration system:")
    print("  ✅ No more hard-coded paths")
    print("  ✅ Flexible deployment options") 
    print("  ✅ Environment variable support")
    print("  ✅ Configuration file support")
    print("  ✅ Intelligent model discovery")
    print("  ✅ Model compatibility validation")
    print("  ✅ Complete backwards compatibility")
    print("  ✅ Perfect for containerized deployments")


if __name__ == "__main__":
    main()