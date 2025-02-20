# zero_copy_GPU
Zero copy data transfer from client via commodity network to NVIDIA GPUs using zero copy TCP.

## Requirements
Requires Linux Kernel >= 4.18
CONFIG_TCP_ZERO_COPY_TRANSFER_COMPLETION_NOTIFICATION must be on
First run check_zerocopy.sh

How to compile
-----------------
make


test with ''nsys nvprof ./receive''
vs
nsys nvprof ''./receive -u''

Performance gains even if it is not a perfect one (reusing the buffer, no actual copy)
