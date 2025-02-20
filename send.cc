#include "common.h"

int main(int argc, char *argv[]) {
	struct sockaddr_storage addr;
	size_t buffer_sz;
	unsigned long map_align = default_huge_page_size();
	int on = 1;
	static socklen_t cfg_alen = sizeof(struct sockaddr_in6);
	uint64_t total = 0;
	bool zflg = true;
	int c;

	while ((c = getopt(argc, argv, "zras:")) != -1) {
		switch (c) {
			case 'z':
				zflg = false;
				break;
			default:
				fprintf(stderr, "Usage: %s [-z] \n"
						"  -z: Disable TCP zero-copy send (default: on)\n",
						argv[0]);
				exit(1);
		}
	}


	void *buffer = mmap_large_buffer(chunk_size, &buffer_sz, map_align);
	if (buffer == (unsigned char *)-1) {
		perror("mmap");
		exit(1);
	}
	int fd = create_new_socket(addr);
	if (setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &on, sizeof(on)) == -1) {
		perror("setsockopt SO_ZEROCOPY, (-z option disabled)");
		zflg = false;
	}

	std::cout << "Sending " << FILE_SZ << " Bytes ";
	if(zflg){
		std::cout << " with zero copy TCP" << std::endl;
	}else{
		std::cout << " without zero copy TCP" << std::endl;
	}

	if (connect(fd, (const struct sockaddr *)&addr, cfg_alen) == -1) {
		perror("connect");
		exit(1);
	}

	auto start_time = std::chrono::high_resolution_clock::now();

	while (total < FILE_SZ) {
		size_t offset = total % chunk_size;
		int64_t wr = FILE_SZ - total;

		if (wr > chunk_size - offset)
			wr = chunk_size - offset;
		/* Note : we just want to fill the pipe with random bytes */
		wr = send(fd, (uint8_t*)buffer + offset,
				(size_t)wr, zflg ? MSG_ZEROCOPY : 0);
		if (wr <= 0)
			break;
		total += wr;
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

	double throughput = (total * 8.0 / 1000000.0) / (duration.count() / 1000.0); // Mbps
	std::cout << "Throughput: " << throughput << " Mbps" << std::endl;

	close(fd);
	munmap(buffer, buffer_sz);
	return 0;
}
