# zero_copy_GPU
Zero copy data movement from network to NVIDIA GPUs using zero copy TCP in userspace
receive.cu receives data from network and send it to the GPU using zero-copy

How to compile
-----------------
make


test with ''nsys nvprof ./receive''
vs
nsys nvprof ''./receive -u''

Performance gains even if it is not a perfect one (reusing the buffer, no actual copy)
