## Prerequisites, Installation, Compilation, and Execution Instructions

### Prerequisites
Before running the code, ensure the following prerequisites are met:

1. **Hardware Requirements:**
   - **DPDK-Supported NIC:**
     - Ensure your NIC is supported by DPDK (e.g., Intel X710, Mellanox ConnectX-5).
     - Check NIC support using:
       ```bash
       lspci | grep Ethernet
       ```
       Compare the output with the DPDK supported NICs list on the [DPDK website](https://core.dpdk.org/supported/).
   - **CUDA-Capable GPU:**
     - Ensure you have an NVIDIA GPU with CUDA support (e.g., Tesla, Quadro, GeForce with CUDA cores).
   - **CPU and Memory:**
     - Multi-core CPU (e.g., 4+ cores) for DPDK polling and application logic.
     - Sufficient RAM (e.g., 8+ GB) with huge pages configured (e.g., 2 MB or 1 GB pages).

2. **Software Requirements:**
   - **Operating System:**
     - Linux distribution with kernel version 4.14 or later (e.g., Ubuntu 20.04, RHEL 8).
   - **DPDK:**
     - Version 21.11 or later recommended for external memory support.
   - **CUDA Toolkit:**
     - Version 11.x or later for pinned memory and asynchronous transfers.
   - **Development Tools:**
     - GCC or Clang with C++11 support.
     - Build tools (e.g., `make`, `meson`, `ninja`).

3. **Kernel Configuration:**
   - Enable IOMMU in the kernel and BIOS:
     - For Intel CPUs: Enable VT-d in BIOS and add `intel_iommu=on` to GRUB.
     - For AMD CPUs: Enable AMD-Vi in BIOS and add `amd_iommu=on` to GRUB.
     - Update GRUB:
       ```bash
       sudo nano /etc/default/grub
       # Add intel_iommu=on or amd_iommu=on to GRUB_CMDLINE_LINUX
       sudo update-grub
       sudo reboot
       ```
   - Verify IOMMU is enabled:
       ```bash
       dmesg | grep -i iommu
       ```

4. **Huge Pages Configuration:**
   - Configure huge pages for DPDK memory management:
       ```bash
       echo 1024 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
       sudo mkdir /mnt/huge
       sudo mount -t hugetlbfs nodev /mnt/huge
       ```
   - Verify huge pages:
       ```bash
       grep Huge /proc/meminfo
       ```

---

### Installation

1. **Install DPDK:**
   - Download and install DPDK (e.g., version 21.11):
       ```bash
       wget https://fast.dpdk.org/rel/dpdk-21.11.tar.xz
       tar -xvf dpdk-21.11.tar.xz
       cd dpdk-21.11
       meson build
       ninja -C build
       sudo ninja -C build install
       ```
  - Set environment variables:
       ```bash
       export RTE_SDK=/path/to/dpdk-21.11
       export RTE_TARGET=x86_64-native-linux-gcc
       ```
   - Load DPDK kernel modules:
       ```bash
       sudo modprobe vfio-pci
       sudo modprobe uio
      sudo insmod $RTE_SDK/build/kmod/igb_uio.ko
       ```
   - Bind the NIC to `vfio-pci` or `igb_uio`:
     - Find the NIC's PCI device ID:
       ```bash
       lspci | grep Ethernet
       ```
       Example output:
       ```
       03:00.0 Ethernet controller: Intel Corporation 82599ES 10-Gigabit SFI/SFP+ Network Connection (rev 01)
       ```
     - Bind the NIC:
       ```bash
       sudo dpdk-devbind.py --bind=vfio-pci 03:00.0
       ```
     - Verify binding:
       ```bash
       dpdk-devbind.py --status
       ```

2. **Install CUDA Toolkit:**
   - Download and install the CUDA Toolkit (e.g., version 11.8) from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
   - For Ubuntu 20.04:
       ```bash
       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
       sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
       sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
       sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
       sudo apt-get update
       sudo apt-get install cuda
      ```
   - Add CUDA to your PATH:
       ```bash
       export PATH=/usr/local/cuda-11.8/bin:$PATH
       export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
       ```
   - Verify CUDA installation:
       ```bash
       nvcc --version
       nvidia-smi
       ```

3. **Install Development Tools:**
   - Install GCC, Clang, and build tools:
       ```bash
       sudo apt-get install build-essential meson ninja-build
       ```

---

### Compile the Code

1. **Save the Code:**
   - Save the receiver code from the previous response as `dpdk_cuda_receiver_zero_copy.cc`.

2. **Compile the Code:**
   - Compile with DPDK and CUDA libraries:
       ```bash
       g++ -O3 -I/usr/local/include -I/usr/local/cuda/include \
           -L/usr/local/lib -L/usr/local/cuda/lib64 \
           -o dpdk_cuda_receiver_zero_copy dpdk_cuda_receiver_zero_copy.cc \
           -ldpdk -lcudart -lpthread -ldl -lnuma
       ```
   - Notes:
     - Adjust include and library paths if DPDK or CUDA are installed in non-standard locations.
     - Ensure `LD_LIBRARY_PATH` includes DPDK and CUDA library paths:
       ```bash
       export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
       ```

---

### Run the Code

1. **Run as Root:**
   - DPDK requires root privileges for NIC access and memory management.
   - Run the receiver:
       ```bash
       sudo ./dpdk_cuda_receiver_zero_copy --lcores=0-1 -- -p 0x1
       ```
   - Options:
     - `--lcores=0-1`: Specifies CPU cores 0 and 1 for DPDK.
     - `-p 0x1`: Enables port 0 (adjust if using a different port).

2. **Expected Output:**
   - The code will:
     - Initialize DPDK and configure the NIC.
     - Allocate CUDA pinned memory and register it with DPDK.
     - Receive packets directly into the pinned buffer (zero-copy).
     - Transfer data to GPU memory.
     - Perform cleanup and exit.
   - Example output:
       ```
       Found 1 Ethernet ports
       Allocated CUDA pinned memory at 0x7f1234567890
       Registered CUDA pinned memory with DPDK
       Created mbuf pool with external CUDA pinned memory
       Allocated GPU memory at 0x7f9876543210
       Received 1048576 bytes of data
       Started asynchronous transfer to GPU
       GPU transfer completed
       Cleanup completed successfully
       ```
