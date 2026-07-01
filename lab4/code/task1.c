#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

static double a, b, c;
static double b_sq, four_ac;
static int b_sq_done = 0, four_ac_done = 0;
static double x1, x2;
static int n_real_roots;

static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  cond = PTHREAD_COND_INITIALIZER;

static void *compute_b_sq(void *arg) {
    (void)arg;
    double val = b * b;
    pthread_mutex_lock(&mtx);
    b_sq = val;
    b_sq_done = 1;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mtx);
    return NULL;
}

static void *compute_four_ac(void *arg) {
    (void)arg;
    double val = 4.0 * a * c;
    pthread_mutex_lock(&mtx);
    four_ac = val;
    four_ac_done = 1;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mtx);
    return NULL;
}

static void *compute_roots(void *arg) {
    (void)arg;
    pthread_mutex_lock(&mtx);
    while (!b_sq_done || !four_ac_done) {
        pthread_cond_wait(&cond, &mtx);
    }
    double delta = b_sq - four_ac;
    double den = 2.0 * a;
    if (delta > 0) {
        n_real_roots = 2;
        x1 = (-b + sqrt(delta)) / den;
        x2 = (-b - sqrt(delta)) / den;
    } else if (delta == 0) {
        n_real_roots = 1;
        x1 = -b / den;
    } else {
        n_real_roots = 0;
    }
    pthread_mutex_unlock(&mtx);
    return NULL;
}

int main(void) {
    printf("请输入一元二次方程 ax^2+bx+c=0 的系数 a, b, c: ");
    if (scanf("%lf %lf %lf", &a, &b, &c) != 3) {
        fprintf(stderr, "输入格式错误\n");
        return 1;
    }
    if (a == 0) {
        fprintf(stderr, "a 不能为 0\n");
        return 1;
    }

    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    pthread_t t1, t2, t3;
    pthread_create(&t1, NULL, compute_b_sq, NULL);
    pthread_create(&t2, NULL, compute_four_ac, NULL);
    pthread_create(&t3, NULL, compute_roots, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);

    gettimeofday(&tv_end, NULL);
    double elapsed = (tv_end.tv_sec - tv_start.tv_sec)
                   + (tv_end.tv_usec - tv_start.tv_usec) / 1e6;

    printf("方程 %.6g*x^2 + %.6g*x + %.6g = 0\n", a, b, c);
    if (n_real_roots == 2) {
        printf("有两个实数根: x1 = %.6g, x2 = %.6g\n", x1, x2);
    } else if (n_real_roots == 1) {
        printf("有一个重根: x = %.6g\n", x1);
    } else {
        printf("无实数根\n");
    }
    printf("求解耗时: %.6f 秒\n", elapsed);

    pthread_mutex_destroy(&mtx);
    pthread_cond_destroy(&cond);
    return 0;
}
