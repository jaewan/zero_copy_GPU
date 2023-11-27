#define _GNU_SOURCE
#include <sys/types.h>
#include <fcntl.h>
#include <error.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <poll.h>
#include <linux/tcp.h>
#include <assert.h>

#ifndef MSG_ZEROCOPY
#define MSG_ZEROCOPY    0x4000000
#endif

#ifndef min
#define min(a, b)  ((a) < (b) ? (a) : (b))
#endif

#define MSS 4108
#define FILE_SZ (1ULL << 35)

static size_t map_align;
static int cfg_family = AF_INET6;
static socklen_t cfg_alen = sizeof(struct sockaddr_in6);
static size_t chunk_size  = 512*1024;
int *device_array;
unsigned int size;

static void *mmap_large_buffer(size_t need, size_t *allocated)
{
	void *buffer;
	size_t sz;

	/* Attempt to use huge pages if possible. */
	sz = ALIGN_UP(need, map_align);
	buffer = mmap(NULL, sz, PROT_READ | PROT_WRITE,
		      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

	if (buffer == (void *)-1) {
		sz = need;
		buffer = mmap(NULL, sz, PROT_READ | PROT_WRITE,
			      MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
			      -1, 0);
		if (buffer != (void *)-1)
			fprintf(stderr, "MAP_HUGETLB attempt failed, look at /sys/kernel/mm/hugepages for optimal performance\n");
	}
	*allocated = sz;
	return buffer;
}

void mmap_and_read(int fd)
{
	unsigned long total_mmap = 0, total = 0;
	struct tcp_zerocopy_receive zc;
	unsigned char *buffer = NULL;
	unsigned long delta_usec;
	EVP_MD_CTX *ctx = NULL;
	struct timeval t0, t1;
	void *raddr = NULL;
	void *addr = NULL;
	double throughput;
	struct rusage ru;
	size_t buffer_sz;
	int lu, zflg = 1;
	const unsigned int N = 1048576;
	const unsigned int bytes = N * sizeof(int);

	gettimeofday(&t0, NULL);

	fcntl(fd, F_SETFL, O_NDELAY);
	buffer = (unsigned char*)mmap_large_buffer(chunk_size, &buffer_sz);

	if (buffer == (void *)-1) {
		perror("mmap");
		goto error;
	}
	raddr = mmap(NULL, chunk_size + map_align, PROT_READ, MAP_SHARED, fd, 0);
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
			zc.length = min(chunk_size, FILE_SZ - total);

			res = getsockopt(fd, IPPROTO_TCP, TCP_ZEROCOPY_RECEIVE,
					 &zc, &zc_len);
			if (res == -1)
				break;

			if (zc.length) {
				assert(zc.length <= chunk_size);
				total_mmap += zc.length;
				/* It is more efficient to unmap the pages right now,
				 * instead of doing this in next TCP_ZEROCOPY_RECEIVE.
				 */
				cudaMemcpy(device_array + total, addr, zc.length, cudaMemcpyHostToDevice);
				madvise(addr, zc.length, MADV_DONTNEED);
				total += zc.length;
			}
			if (zc.recv_skip_hint) {
				assert(zc.recv_skip_hint <= chunk_size);
				int read_size = min(zc.recv_skip_hint, FILE_SZ - total)
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
		while (sub < chunk_size) {
			int read_size = min(chunk_size - sub, FILE_SZ - total);
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
		munmap(raddr, chunk_size + map_align);
	return;
}

static unsigned long default_huge_page_size(void)
{
	FILE *f = fopen("/proc/meminfo", "r");
	unsigned long hps = 0;
	size_t linelen = 0;
	char *line = NULL;

	if (!f)
		return 0;
	while (getline(&line, &linelen, f) > 0) {
		if (sscanf(line, "Hugepagesize:       %lu kB", &hps) == 1) {
			hps <<= 10;
			break;
		}
	}
	free(line);
	fclose(f);
	return hps;
}

static void do_accept(int fdlisten)
{
	int rcvlowat = chunk_size;
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
		continue;
	}
	mmap_and_read(fd);
}

// param bytes should be set as SIZE * sizeof(typeof(SIZE))
// If its type is not int, change its type here and global size variable as well
void receive(const unsigned int bytes){
	int on = 1;
	char *host = NULL;
	struct sockaddr_storage listenaddr, addr;
	int mss = MSS;

	// Allocating array in the device
	cudaMalloc((int**)&device_array, bytes);
	size = bytes

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
	do_accept(fdlisten, bytes);
}
