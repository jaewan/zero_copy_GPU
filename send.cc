#include "common.h"

#define SEND_SIZE (1UL << 35)


int main(){
	char *host = NULL;
	struct sockaddr_storage addr;
	size_t buffer_sz;
	unsigned long map_align = default_huge_page_size();
	int mss = MSS, on = 1;
	static int cfg_family = AF_INET6;
	static socklen_t cfg_alen = sizeof(struct sockaddr_in6);
	uint64_t total = 0;
	int zflg = 1;

	void *buffer = mmap_large_buffer(chunk_size, &buffer_sz, map_align);
	if (buffer == (unsigned char *)-1) {
		perror("mmap");
		exit(1);
	}

	int fd = socket(cfg_family, SOCK_STREAM, 0);
	if (fd == -1) {
		perror("socket");
		exit(1);
	}

	setup_sockaddr(cfg_family, host, &addr);

	if (mss &&
	    setsockopt(fd, IPPROTO_TCP, TCP_MAXSEG, &mss, sizeof(mss)) == -1) {
		perror("setsockopt TCP_MAXSEG");
		exit(1);
	}
	if (connect(fd, (const struct sockaddr *)&addr, cfg_alen) == -1) {
		perror("connect");
		exit(1);
	}

	if (setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY,
			       &on, sizeof(on)) == -1) {
		perror("setsockopt SO_ZEROCOPY, (-z option disabled)");
		zflg = 0;
	}

	while (total < SEND_SIZE) {
		size_t offset = total % chunk_size;
		int64_t wr = SEND_SIZE - total;

		if (wr > chunk_size - offset)
			wr = chunk_size - offset;
		/* Note : we just want to fill the pipe with random bytes */
		wr = send(fd, buffer + offset,
			  (size_t)wr, zflg ? MSG_ZEROCOPY : 0);
		if (wr <= 0)
			break;
		total += wr;
	}
	close(fd);
	munmap(buffer, buffer_sz);
	return 0;
}
