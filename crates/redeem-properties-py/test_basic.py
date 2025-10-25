#!/usr/bin/env python3
"""
Simple test script to verify the Python bindings work correctly.
"""

import sys

def test_imports():
    """Test that the module can be imported and classes are available."""
    print("Testing imports...")
    try:
        import redeem_properties_py
        print("✓ Module imported successfully")
        
        # Check that classes exist
        assert hasattr(redeem_properties_py, 'RTModel'), "RTModel class not found"
        print("✓ RTModel class found")
        
        assert hasattr(redeem_properties_py, 'CCSModel'), "CCSModel class not found"
        print("✓ CCSModel class found")
        
        assert hasattr(redeem_properties_py, 'MS2Model'), "MS2Model class not found"
        print("✓ MS2Model class found")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_module_docstring():
    """Test that docstrings are accessible."""
    print("\nTesting module docstrings...")
    try:
        import redeem_properties_py
        
        # Check module docstring
        if redeem_properties_py.__doc__:
            print(f"✓ Module docstring: {redeem_properties_py.__doc__[:80]}...")
        
        # Check class docstrings
        for cls_name in ['RTModel', 'CCSModel', 'MS2Model']:
            cls = getattr(redeem_properties_py, cls_name)
            if cls.__doc__:
                print(f"✓ {cls_name} docstring: {cls.__doc__[:80]}...")
        
        return True
    except Exception as e:
        print(f"✗ Docstring test failed: {e}")
        return False

def test_error_handling():
    """Test that appropriate errors are raised for invalid inputs."""
    print("\nTesting error handling...")
    try:
        import redeem_properties_py
        
        # Try to create a model with an invalid path
        try:
            model = redeem_properties_py.RTModel(
                "/nonexistent/path.safetensors",
                "rt_cnn_lstm"
            )
            print("✗ Expected error for nonexistent model file")
            return False
        except Exception as e:
            print(f"✓ Got expected error for invalid model path: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ReDeeM Properties Python Bindings - Basic Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_module_docstring()
    all_passed &= test_error_handling()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
