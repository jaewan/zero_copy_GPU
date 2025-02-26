#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/queue.h>

#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#include <cuda_runtime.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32
#define DPDK_PORT_ID 0

// GPU buffer size and data size tracking
#define MAX_PACKET_SIZE 1500
#define GPU_BUFFER_SIZE (MAX_PACKET_SIZE * BURST_SIZE)
static volatile int total_received_bytes = 0;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// DPDK port configuration
static const struct rte_eth_conf port_conf = {
    .rxmode = {
        .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
        .mq_mode = ETH_MQ_RX_RSS,
    },
    .rx_adv_conf = {
        .rss_conf = {
            .rss_key = NULL,
            .rss_hf = ETH_RSS_IP | ETH_RSS_TCP | ETH_RSS_UDP,
        },
    },
};

// Initialize DPDK environment and configure port
static int dpdk_init(struct rte_mempool **mbuf_pool) {
    // Initialize EAL
    int ret = rte_eal_init(0, NULL);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Error initializing EAL\n");

    // Check that at least one port is available
    unsigned nb_ports = rte_eth_dev_count_avail();
    if (nb_ports < 1)
        rte_exit(EXIT_FAILURE, "Error: no Ethernet ports available\n");

    // Create the mbuf pool
    *mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (*mbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    // Configure the Ethernet device
    struct rte_eth_dev_info dev_info;
    ret = rte_eth_dev_info_get(DPDK_PORT_ID, &dev_info);
    if (ret != 0)
        rte_exit(EXIT_FAILURE, "Cannot get device info\n");

    ret = rte_eth_dev_configure(DPDK_PORT_ID, 1, 1, &port_conf);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d\n", ret);

    // Allocate and set up RX queue
    ret = rte_eth_rx_queue_setup(DPDK_PORT_ID, 0, RX_RING_SIZE,
            rte_eth_dev_socket_id(DPDK_PORT_ID), NULL, *mbuf_pool);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Cannot set up RX queue: err=%d\n", ret);

    // Allocate and set up TX queue
    ret = rte_eth_tx_queue_setup(DPDK_PORT_ID, 0, TX_RING_SIZE,
            rte_eth_dev_socket_id(DPDK_PORT_ID), NULL);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Cannot set up TX queue: err=%d\n", ret);

    // Start the Ethernet port
    ret = rte_eth_dev_start(DPDK_PORT_ID);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Cannot start device: err=%d\n", ret);

    // Enable RX in promiscuous mode
    rte_eth_promiscuous_enable(DPDK_PORT_ID);

    return 0;
}

// Main processing loop
static int packet_processing_loop(struct rte_mempool *mbuf_pool) {
    printf("Core %u processing packets\n", rte_lcore_id());
    
    // Allocate CUDA pinned memory for packet data
    char *host_buffer;
    CUDA_CHECK(cudaHostAlloc(&host_buffer, GPU_BUFFER_SIZE, 
                            cudaHostAllocMapped | cudaHostAllocWriteCombined));
    
    // Allocate GPU memory
    char *gpu_buffer;
    CUDA_CHECK(cudaMalloc(&gpu_buffer, GPU_BUFFER_SIZE));
    
    // Create CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    printf("Waiting for packets...\n");
    
    while (1) {
        // Receive a burst of packets
        struct rte_mbuf *mbufs[BURST_SIZE];
        unsigned num_rx = rte_eth_rx_burst(DPDK_PORT_ID, 0, mbufs, BURST_SIZE);
        
        if (num_rx > 0) {
            int batch_size = 0;
            
            // Process received packets
            for (unsigned i = 0; i < num_rx; i++) {
                // Get packet data pointer and length
                char *pkt_data = rte_pktmbuf_mtod(mbufs[i], char *);
                unsigned pkt_len = rte_pktmbuf_data_len(mbufs[i]);
                
                // Skip Ethernet header (typically 14 bytes)
                // Adjust this depending on your protocol
                char *payload = pkt_data + 14;
                unsigned payload_len = pkt_len - 14;
                
                // Copy payload to the pinned buffer
                // In a production system, you might want to do zero-copy
                // by using rte_mbuf with CUDA directly
                memcpy(host_buffer + batch_size, payload, payload_len);
                batch_size += payload_len;
                
                // Free the mbuf
                rte_pktmbuf_free(mbufs[i]);
            }
            
            // Transfer batch of data to GPU
            CUDA_CHECK(cudaMemcpyAsync(gpu_buffer, host_buffer, batch_size, 
                                     cudaMemcpyHostToDevice, stream));
            
            // Process data on GPU (would add kernel launch here)
            // kernel<<<grid, block, 0, stream>>>(gpu_buffer, batch_size);
            
            // Synchronize to ensure transfer is complete before next batch
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            total_received_bytes += batch_size;
            printf("Transferred %d bytes to GPU (total: %d)\n", 
                  batch_size, total_received_bytes);
        }
    }
    
    // Cleanup (never reached in this infinite loop example)
    CUDA_CHECK(cudaFreeHost(host_buffer));
    CUDA_CHECK(cudaFree(gpu_buffer));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return 0;
}

int main(int argc, char **argv) {
    struct rte_mempool *mbuf_pool;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Initialize DPDK
    if (dpdk_init(&mbuf_pool) < 0)
        return -1;
    
    // Start packet processing on the main core
    packet_processing_loop(mbuf_pool);
    
    return 0;
}
