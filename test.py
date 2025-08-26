import torch

# --- 1. Check if PyTorch can see CUDA devices ---
# This is the most basic check.
is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {is_cuda_available}")

if not is_cuda_available:
    print("No CUDA devices detected. Please ensure your drivers and environment are set up correctly.")
    exit()

# --- 2. Get the total number of visible GPUs ---
# This number is affected by the CUDA_VISIBLE_DEVICES environment variable.
device_count = torch.cuda.device_count()
print(f"Total number of visible GPUs: {device_count}")

# --- 3. Iterate through each visible GPU and print its properties ---
# This will show you details about each device, identified by its index (0, 1, 2, ...).
print("\n--- Listing properties for each visible GPU ---")
for i in range(device_count):
    # Select the device by its index
    torch.cuda.set_device(i)

    # Get the device properties
    device_properties = torch.cuda.get_device_properties(i)

    print(f"\n--- GPU {i} ---")
    print(f"  Name: {device_properties.name}")
    print(f"  CUDA Compute Capability: {device_properties.major}.{device_properties.minor}")
    
    # Convert memory from bytes to GB for readability
    total_memory_gb = device_properties.total_memory / (1024**3)
    print(f"  Total Memory: {total_memory_gb:.2f} GB")
    
    # You can get more details as needed
    print(f"  Number of Streaming Multiprocessors (SMs): {device_properties.multi_processor_count}")
    
    # Check if this GPU is a MIG device
    # Note: PyTorch doesn't have a direct function for this, but the device name often indicates it.
    if "MIG" in device_properties.name:
        print("  This appears to be a MIG device.")