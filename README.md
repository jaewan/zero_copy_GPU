# zero_copy_GPU
Zero copy data movement from network to NVIDIA GPUs using zero copy TCP in userspace

compile with make

test with nsys nvprof ./receive 
vs
nsys nvprof ./receive -u 

Performance gains even if it is not a perfect one (reusing the buffer, no actual copy)
