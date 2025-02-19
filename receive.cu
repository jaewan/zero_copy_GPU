#include "common.h"

// Configuration flags for different optimization techniques
struct Config {
	bool use_zerocopy_tcp;      // Use TCP zerocopy receive
	bool use_cuda_register;     // Use cudaHostRegister
	bool use_async_transfer;    // Use cudaMemcpyAsync with streams
	int num_streams;            // Number of CUDA streams to use
} config;

// Statistics structure
struct Statistics {
	unsigned long total_bytes;
	unsigned long total_mmap_bytes;
	struct timeval start_time;
	struct timeval end_time;
	struct rusage ru;
} stats;

static size_t map_align;
int *device_array;
unsigned long long size;
cudaStream_t* streams = NULL;

// Helper functions
static void checkCuda(cudaError_t result, const char *func) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA error in %s: %s\n", func, cudaGetErrorString(result));
		exit(1);
	}
}

static void init_streams() {
	if (config.use_async_transfer && config.num_streams > 0) {
		streams = (cudaStream_t*)malloc(config.num_streams * sizeof(cudaStream_t));
		for(int i = 0; i < config.num_streams; i++) {
			checkCuda(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
		}
	}
}

static void cleanup_streams() {
	if (streams) {
		for(int i = 0; i < config.num_streams; i++) {
			cudaStreamDestroy(streams[i]);
		}
		free(streams);
		streams = NULL;
	}
}

static void print_statistics(const char* test_name) {
	unsigned long delta_usec = (stats.end_time.tv_sec - stats.start_time.tv_sec) * 1000000 +
		stats.end_time.tv_usec - stats.start_time.tv_usec;
	double throughput = 0;
	if (delta_usec) {
		throughput = stats.total_bytes * 8.0 / (double)delta_usec / 1000.0;
	}

	printf("\n=== Test Configuration: %s ===\n", test_name);
	printf("Zero-copy TCP: %s\n", config.use_zerocopy_tcp ? "ON" : "OFF");
	printf("CUDA Register: %s\n", config.use_cuda_register ? "ON" : "OFF");
	printf("Async Transfer: %s\n", config.use_async_transfer ? "ON" : "OFF");
	if (config.use_async_transfer)
		printf("Number of streams: %d\n", config.num_streams);

	printf("\n=== Performance Results ===\n");
	printf("Total data: %lg MB\n", stats.total_bytes / (1024.0 * 1024.0));
	if (config.use_zerocopy_tcp)
		printf("Mmap'ed data: %lg MB (%lg%%)\n",
				stats.total_mmap_bytes / (1024.0 * 1024.0),
				100.0 * stats.total_mmap_bytes / stats.total_bytes);
	printf("Transfer time: %lg seconds\n", (double)delta_usec / 1000000.0);
	printf("Throughput: %lg Gbps\n", throughput);

	unsigned long total_usec = 1000000 * stats.ru.ru_utime.tv_sec + stats.ru.ru_utime.tv_usec +
		1000000 * stats.ru.ru_stime.tv_sec + stats.ru.ru_stime.tv_usec;
	unsigned long mb = stats.total_bytes >> 20;
	printf("CPU usage - User: %lg s, System: %lg s\n",
			(double)stats.ru.ru_utime.tv_sec + (double)stats.ru.ru_utime.tv_usec / 1000000.0,
			(double)stats.ru.ru_stime.tv_sec + (double)stats.ru.ru_stime.tv_usec / 1000000.0);
	printf("Context switches: %lu\n", stats.ru.ru_nvcsw);
	printf("%lg usec per MB\n", (double)total_usec/mb);
	printf("=============================\n\n");
}

void mmap_and_read(int fd)
{
	void *raddr = NULL;
	void *addr = NULL;
	void *device_ptr = NULL;
	unsigned char *buffer = NULL;
	size_t buffer_sz;
	int lu;
	bool using_registered_memory = false;

	// Initialize statistics
	memset(&stats, 0, sizeof(stats));
	gettimeofday(&stats.start_time, NULL);

	// Initialize streams if needed
	init_streams();

	fcntl(fd, F_SETFL, O_NDELAY);
	buffer = (unsigned char*)mmap_large_buffer(CHUNK_SIZE, &buffer_sz, map_align);
	if (buffer == (void *)-1) {
		perror("mmap");
		goto error;
	}

	if (config.use_zerocopy_tcp) {
		raddr = mmap(NULL, CHUNK_SIZE + map_align, PROT_READ, MAP_SHARED, fd, 0);
		if (raddr == (void *)-1) {
			perror("mmap");
			config.use_zerocopy_tcp = false;
		} else {
			addr = ALIGN_PTR_UP(raddr, map_align);

			if (config.use_cuda_register) {
				cudaError_t reg_result = cudaHostRegister(addr, CHUNK_SIZE, cudaHostRegisterDefault);
				if (reg_result == cudaSuccess) {
					checkCuda(cudaHostGetDevicePointer(&device_ptr, addr, 0),
							"cudaHostGetDevicePointer");
					using_registered_memory = true;
				} else {
					fprintf(stderr, "Warning: CUDA memory registration failed\n");
				}
			}
		}
	}

	while (1) {
		if (config.use_zerocopy_tcp) {
			struct tcp_zerocopy_receive zc;
			socklen_t zc_len = sizeof(zc);

			memset(&zc, 0, sizeof(zc));
			zc.address = (__u64)((unsigned long)addr);
			zc.length = min(CHUNK_SIZE, size - stats.total_bytes);

			if (getsockopt(fd, IPPROTO_TCP, TCP_ZEROCOPY_RECEIVE, &zc, &zc_len) == -1)
				break;

			if (zc.length) {
				stats.total_mmap_bytes += zc.length;

				if (using_registered_memory) {
					// Use registered memory
					if (config.use_async_transfer) {
						checkCuda(cudaMemcpyAsync(device_array + stats.total_bytes/sizeof(int),
									(char*)device_ptr + (stats.total_bytes % CHUNK_SIZE),
									zc.length,
									cudaMemcpyHostToDevice,
									streams[stats.total_bytes % config.num_streams]),
								"cudaMemcpyAsync");
					} else {
						checkCuda(cudaMemcpy(device_array + stats.total_bytes/sizeof(int),
									(char*)device_ptr + (stats.total_bytes % CHUNK_SIZE),
									zc.length,
									cudaMemcpyHostToDevice),
								"cudaMemcpy");
					}
				} else {
					// Regular copy
					if (config.use_async_transfer) {
						checkCuda(cudaMemcpyAsync(device_array + stats.total_bytes/sizeof(int),
									addr,
									zc.length,
									cudaMemcpyHostToDevice,
									streams[stats.total_bytes % config.num_streams]),
								"cudaMemcpyAsync");
					} else {
						checkCuda(cudaMemcpy(device_array + stats.total_bytes/sizeof(int),
									addr,
									zc.length,
									cudaMemcpyHostToDevice),
								"cudaMemcpy");
					}
				}

				madvise(addr, zc.length, MADV_DONTNEED);
				stats.total_bytes += zc.length;
			}

			if (zc.recv_skip_hint) {
				int read_size = min(zc.recv_skip_hint, size - stats.total_bytes);
				lu = read(fd, buffer, read_size);
				if (lu > 0) {
					if (config.use_async_transfer) {
						checkCuda(cudaMemcpyAsync(device_array + stats.total_bytes/sizeof(int),
									buffer,
									lu,
									cudaMemcpyHostToDevice,
									streams[stats.total_bytes % config.num_streams]),
								"cudaMemcpyAsync");
					} else {
						checkCuda(cudaMemcpy(device_array + stats.total_bytes/sizeof(int),
									buffer,
									lu,
									cudaMemcpyHostToDevice),
								"cudaMemcpy");
					}
					stats.total_bytes += lu;
				}
				if (lu == 0)
					goto end;
			}
		} else {
			// Regular read path
			lu = read(fd, buffer, min(CHUNK_SIZE, size - stats.total_bytes));
			if (lu <= 0)
				goto end;

			if (config.use_async_transfer) {
				checkCuda(cudaMemcpyAsync(device_array + stats.total_bytes/sizeof(int),
							buffer,
							lu,
							cudaMemcpyHostToDevice,
							streams[stats.total_bytes % config.num_streams]),
						"cudaMemcpyAsync");
			} else {
				checkCuda(cudaMemcpy(device_array + stats.total_bytes/sizeof(int),
							buffer,
							lu,
							cudaMemcpyHostToDevice),
						"cudaMemcpy");
			}
			stats.total_bytes += lu;
		}
	}

end:
	// Synchronize all streams if using async transfer
	if (config.use_async_transfer) {
		for(int i = 0; i < config.num_streams; i++) {
			checkCuda(cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize");
		}
	}

	gettimeofday(&stats.end_time, NULL);
	getrusage(RUSAGE_THREAD, &stats.ru);

	// Generate test name based on configuration
	char test_name[256];
	snprintf(test_name, sizeof(test_name), "TCP%s_CUDA%s_%s_%d",
			config.use_zerocopy_tcp ? "_ZEROCOPY" : "",
			config.use_cuda_register ? "_REGISTER" : "",
			config.use_async_transfer ? "ASYNC" : "SYNC",
			config.use_async_transfer ? config.num_streams : 0);

	print_statistics(test_name);

error:
	if (using_registered_memory) {
		cudaHostUnregister(addr);
	}
	cleanup_streams();
	munmap(buffer, buffer_sz);
	close(fd);
	if (config.use_zerocopy_tcp && raddr)
		munmap(raddr, CHUNK_SIZE + map_align);
}


static void do_accept(int fdlisten) {
    int rcvlowat = CHUNK_SIZE;
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    int fd;
    if (setsockopt(fdlisten, SOL_SOCKET, SO_RCVLOWAT, &rcvlowat, sizeof(rcvlowat)) == -1) {
        perror("setsockopt SO_RCVLOWAT");
    }

    fd = accept(fdlisten, (struct sockaddr *)&addr, &addrlen);
    if (fd == -1) {
        perror("accept");
        exit(1);
    }
    mmap_and_read(fd);
}

void receive(const unsigned long long bytes) {
    int on = 1;
    char *host = NULL;
    struct sockaddr_storage listenaddr;
    int mss = MSS;
    static int cfg_family = AF_INET6;
    static socklen_t cfg_alen = sizeof(struct sockaddr_in6);

    // Allocate array in the device
    checkCuda(cudaMalloc((int **)&device_array, bytes), "Allocating GPU memory");
    size = bytes;

    map_align = default_huge_page_size();
    if (!map_align)
        map_align = 2 * 1024 * 1024;

    int fdlisten = socket(cfg_family, SOCK_STREAM, 0);
    if (fdlisten == -1) {
        perror("socket");
        exit(1);
    }
    setsockopt(fdlisten, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    setup_sockaddr(cfg_family, host, &listenaddr);

    if (mss &&
        setsockopt(fdlisten, IPPROTO_TCP, TCP_MAXSEG, &mss, sizeof(mss)) == -1) {
        perror("setsockopt TCP_MAXSEG");
        exit(1);
    }
    if (bind(fdlisten, (const struct sockaddr *)&listenaddr, cfg_alen) == -1) {
        perror("bind");
        exit(1);
    }
    if (listen(fdlisten, 128) == -1) {
        perror("listen");
        exit(1);
    }
    do_accept(fdlisten);
}

// Modified main function to support different configurations
int main(int argc, char *argv[]) {
	int c;

	// Default configuration
	config.use_zerocopy_tcp = true;
	config.use_cuda_register = false;
	config.use_async_transfer = false;
	config.num_streams = 4;

	while ((c = getopt(argc, argv, "zras:")) != -1) {
		switch (c) {
			case 'z':
				config.use_zerocopy_tcp = true;
				break;
			case 'r':
				config.use_cuda_register = true;
				break;
			case 'a':
				config.use_async_transfer = true;
				break;
			case 's':
				config.num_streams = atoi(optarg);
				break;
			default:
				fprintf(stderr, "Usage: %s [-z] [-r] [-a] [-s num_streams]\n"
						"  -z: Enable TCP zero-copy (default: on)\n"
						"  -r: Enable CUDA memory registration\n"
						"  -a: Enable async transfer\n"
						"  -s: Number of CUDA streams (default: 4)\n",
						argv[0]);
				exit(1);
		}
	}

	receive(FILE_SZ);
	return 0;
}
