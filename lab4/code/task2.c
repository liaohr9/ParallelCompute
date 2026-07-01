#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>

#define NUM_THREADS 4

static int total_n;
static long long hits_in_circle[NUM_THREADS];
static unsigned int seeds[NUM_THREADS];
static int run_counter = 0;

static void *monte_carlo(void *arg) {
    int id = *(int *)arg;
    int per_thread = total_n / NUM_THREADS;
    int remainder = total_n % NUM_THREADS;
    if (id < remainder) per_thread++;

    unsigned int seed = seeds[id];
    long long local_hits = 0;

    for (int i = 0; i < per_thread; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0) {
            local_hits++;
        }
    }
    hits_in_circle[id] = local_hits;
    return NULL;
}

int main(void) {
    printf("请输入采样点数量 n [1024, 65536]: ");
    if (scanf("%d", &total_n) != 1 || total_n < 1024 || total_n > 65536) {
        fprintf(stderr, "输入无效，n 必须在 [1024, 65536] 范围内\n");
        return 1;
    }

    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    memset(hits_in_circle, 0, sizeof(hits_in_circle));

    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        seeds[i] = (unsigned int)(tv.tv_sec * 1000000ULL + tv.tv_usec
                    + run_counter * 1000000 + i * 7919);
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, monte_carlo, &thread_ids[i]);
    }
    run_counter++;

    long long total_hits = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_hits += hits_in_circle[i];
    }

    gettimeofday(&tv_end, NULL);
    double elapsed = (tv_end.tv_sec - tv_start.tv_sec)
                   + (tv_end.tv_usec - tv_start.tv_usec) / 1e6;

    double pi_approx = 4.0 * (double)total_hits / total_n;

    printf("总采样点数 n     = %d\n", total_n);
    printf("圆内点数 m        = %lld\n", total_hits);
    printf("π 的近似值       = %.10f\n", pi_approx);
    printf("真实 π 值        = %.10f\n", M_PI);
    printf("绝对误差          = %.10f\n", fabs(pi_approx - M_PI));
    printf("求解耗时         = %.6f 秒\n", elapsed);

    return 0;
}
