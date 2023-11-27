#include "common.h"


static size_t map_align;
int *device_array;
unsigned long long size;
int copy_from_user_buffer = 0;

static uint32_t tcp_info_get_rcv_mss(int fd)
{
	socklen_t sz = sizeof(struct tcp_info);
	struct tcp_info info;

	if (getsockopt(fd, IPPROTO_TCP, TCP_INFO, &info, &sz)) {
		fprintf(stderr, "Error fetching TCP_INFO\n");
		return 0;
	}

	return info.tcpi_rcv_mss;
}

void mmap_and_read(int fd)
{
	unsigned long total_mmap = 0, total = 0;
	struct tcp_zerocopy_receive zc;
	unsigned char *buffer = NULL;
	unsigned long delta_usec;
	struct timeval t0, t1;
	void *raddr = NULL;
	void *addr = NULL;
	double throughput;
	struct rusage ru;
	size_t buffer_sz;
	int lu, zflg = 1;

	gettimeofday(&t0, NULL);

	fcntl(fd, F_SETFL, O_NDELAY);
	buffer = (unsigned char*)mmap_large_buffer(CHUNK_SIZE, &buffer_sz, map_align);

	if (buffer == (void *)-1) {
		perror("mmap");
		goto error;
	}
	raddr = mmap(NULL, CHUNK_SIZE + map_align, PROT_READ, MAP_SHARED, fd, 0);
	if (raddr == (void *)-1) {
		perror("mmap");
		zflg = 0;
	} else {
		addr = ALIGN_PTR_UP(raddr, map_align);
	}

	while (1) {
		struct pollfd pfd = { .fd = fd, .events = POLLIN, };
		int sub;

		poll(&pfd, 1, 10000);
		if (zflg) {
			socklen_t zc_len = sizeof(zc);
			int res;

			memset(&zc, 0, sizeof(zc));
			zc.address = (__u64)((unsigned long)addr);
			zc.length = min(CHUNK_SIZE, size - total);

			res = getsockopt(fd, IPPROTO_TCP, TCP_ZEROCOPY_RECEIVE,
					 &zc, &zc_len);
			if (res == -1)
				break;

			if (zc.length) {
				assert(zc.length <= CHUNK_SIZE);
				total_mmap += zc.length;
				/* It is more efficient to unmap the pages right now,
				 * instead of doing this in next TCP_ZEROCOPY_RECEIVE.
				 */
				if (copy_from_user_buffer)
					cudaMemcpy(device_array + total, buffer, zc.length, cudaMemcpyHostToDevice);
				else
					cudaMemcpy(device_array + total, addr, zc.length, cudaMemcpyHostToDevice);
				madvise(addr, zc.length, MADV_DONTNEED);
				total += zc.length;
			}
			if (zc.recv_skip_hint) {
				assert(zc.recv_skip_hint <= CHUNK_SIZE);
				int read_size = min(zc.recv_skip_hint, size - total);
				lu = read(fd, buffer, read_size);
				cudaMemcpy(device_array + total, buffer, read_size, cudaMemcpyHostToDevice);
				if (lu > 0) 
					total += lu;
				if (lu == 0)
					goto end;
			}
			continue;
		}
		sub = 0;
		while (sub < CHUNK_SIZE) {
			int read_size = min(CHUNK_SIZE - sub, size - total);
			lu = read(fd, buffer + sub, read_size);
			cudaMemcpy(device_array + total, buffer, read_size, cudaMemcpyHostToDevice);
			if (lu == 0)
				goto end;
			if (lu < 0)
				break;
			total += lu;
			sub += lu;
		}
	}
end:
	gettimeofday(&t1, NULL);
	delta_usec = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;


	throughput = 0;
	if (delta_usec)
		throughput = total * 8.0 / (double)delta_usec / 1000.0;
	getrusage(RUSAGE_THREAD, &ru);
	if (total > 1024*1024) {
		unsigned long total_usec;
		unsigned long mb = total >> 20;
		total_usec = 1000000*ru.ru_utime.tv_sec + ru.ru_utime.tv_usec +
			     1000000*ru.ru_stime.tv_sec + ru.ru_stime.tv_usec;
		printf("received %lg MB (%lg %% mmap'ed) in %lg s, %lg Gbit\n"
		       "  cpu usage user:%lg sys:%lg, %lg usec per MB, %lu c-switches, rcv_mss %u\n",
				total / (1024.0 * 1024.0),
				100.0*total_mmap/total,
				(double)delta_usec / 1000000.0,
				throughput,
				(double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000.0,
				(double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1000000.0,
				(double)total_usec/mb,
				ru.ru_nvcsw,
				tcp_info_get_rcv_mss(fd));
	}
error:
	munmap(buffer, buffer_sz);
	close(fd);
	if (zflg)
		munmap(raddr, CHUNK_SIZE + map_align);
	return;
}

static void do_accept(int fdlisten)
{
	int rcvlowat = CHUNK_SIZE;
	struct sockaddr_in addr;
	socklen_t addrlen = sizeof(addr);
	int fd;
	if (setsockopt(fdlisten, SOL_SOCKET, SO_RCVLOWAT,
		       &rcvlowat, sizeof(rcvlowat)) == -1) {
		perror("setsockopt SO_RCVLOWAT");
	}

	fd = accept(fdlisten, (struct sockaddr *)&addr, &addrlen);
	if (fd == -1) {
		perror("accept");
		exit(1);
	}
	mmap_and_read(fd);
}

// param bytes should be set as SIZE * sizeof(typeof(SIZE))
// If its type is not int, change its type here and global size variable as well
void receive(const unsigned long long bytes){
	int on = 1;
	char *host = NULL;
	struct sockaddr_storage listenaddr;
	int mss = MSS;
	static int cfg_family = AF_INET6;
	static socklen_t cfg_alen = sizeof(struct sockaddr_in6);

	// Allocating array in the device
	cudaMalloc((int**)&device_array, bytes);
	size = bytes;

	map_align = default_huge_page_size();
	/* if really /proc/meminfo is not helping,
	 * we use the default x86_64 hugepagesize.
	 */
	if (!map_align)
		map_align = 2*1024*1024;
	int fdlisten = socket(cfg_family, SOCK_STREAM, 0);

	if (fdlisten == -1) {
		perror("socket");
		exit(1);
	}
	setsockopt(fdlisten, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

	setup_sockaddr(cfg_family, host, &listenaddr);

	if (mss &&
		setsockopt(fdlisten, IPPROTO_TCP, TCP_MAXSEG,
			   &mss, sizeof(mss)) == -1) {
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

int main(int argc, char *argv[]){
	int c;
	while ((c = getopt(argc, argv, "46p:svr:w:H:zxkP:M:C:a:i:u")) != -1) {
		switch (c) {
		case 'u':
			copy_from_user_buffer = 1;
			break;
		default:
			exit(1);
		}
	}
	receive(FILE_SZ);
	return 0;
}
