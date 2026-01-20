import sys
import numpy as np

def print_header():
    print("\n" + "="*50)
    print("      SWEpy: GPU Acceleration Diagnostic")
    print("="*50)

def print_success():
    print("-" * 50)
    print("\033[92m[✓] SUCCESS: SWEpy is ready for simulation.\033[0m")
    print("    You can now execute 'run_sim.py'.")
    print("=" * 50 + "\n")

def print_error(msg, detail=None):
    print("\n" + "-"*50)
    print(f"\033[91m[X] ERROR: {msg}\033[0m")
    if detail:
        print(f"    Details: {detail}")
    print("="*50 + "\n")

try:
    print_header()

    # 1. Import Libraries
    import cupy as cp
    print(f"[*] CuPy Version Detected: {cp.__version__}")
    
    # 2. Device Handshake & Property Retrieval
    # We use a robust method compatible with both old and new CuPy versions
    dev = cp.cuda.Device(0)
    
    try:
        # Modern CuPy method (Requires decoding bytes)
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        gpu_name = props['name'].decode('utf-8')
    except (AttributeError, KeyError):
        try:
            # Legacy CuPy method
            gpu_name = dev.name
        except AttributeError:
            gpu_name = "Unknown NVIDIA GPU"
    
    print(f"[*] Target GPU: {gpu_name}")
    
    # Optional: Display Compute Capability if available
    try:
        cc = dev.compute_capability
        print(f"[*] Compute Capability: {cc}")
    except:
        pass
    
    # 3. Arithmetic Core Test
    # Performs a simple vector addition to verify memory allocation and kernel execution
    print("[*] Verifying arithmetic kernels...", end=" ")
    
    x_gpu = cp.array([1.0, 2.0, 3.0])
    y_gpu = cp.array([4.0, 5.0, 6.0])
    z_gpu = x_gpu + y_gpu
    
    # 4. Result Validation
    expected = cp.array([5.0, 7.0, 9.0])
    if cp.allclose(z_gpu, expected):
        print("OK")
        print_success()
    else:
        print("FAILED")
        print_error("Arithmetic check failed.", "GPU returned incorrect values.")

except ImportError as e:
    print_error("CuPy library not found.", 
                "Ensure you installed 'cupy-cudaXX' matching your driver.")

except Exception as e:
    print_error("Critical Runtime Error", str(e))
