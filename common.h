#ifndef COMMON_H
#define COMMON_H

#include <sys/mman.h>
#include <fcntl.h>
#include <sys/types.h>
#include <error.h>
#include <sys/socket.h>
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

#define ALIGN_UP(x, align_to)	(((x) + ((align_to)-1)) & ~((align_to)-1))
#define ALIGN_PTR_UP(p, ptr_align_to)	((typeof(p))ALIGN_UP((unsigned long)(p), ptr_align_to))

#define MSS 4108
#define FILE_SZ (1ULL << 35)
#define CFG_PORT 8787
#define CHUNK_SIZE 524288

void *mmap_large_buffer(size_t need, size_t *allocated, size_t map_align);	
unsigned long default_huge_page_size(void);
void setup_sockaddr(int domain, const char *str_addr,
			   struct sockaddr_storage *sockaddr);

// Change this size according to PCIe bandwidth. It is set in common.c
extern size_t chunk_size;
#endif
