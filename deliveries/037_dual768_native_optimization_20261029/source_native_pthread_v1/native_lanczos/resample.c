/* Exact grayscale full-box LANCZOS, derived from Pillow 10.4.0 Resample.c.
 * Copyright (c) 1997-2011 Secret Labs AB; 1995-2011 Fredrik Lundh et al.
 * Copyright (c) 2010-2024 Jeffrey A. Clark and contributors.
 * HPND license is retained in upstream/LICENSE.
 * Changes: CPython capsule entry point, full-box L specialization, independent
 * output-row persistent pthread work sharing. Coefficient and arithmetic retain
 * Pillow's order, 22-bit fixed point, and intermediate uint8 clipping.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
#define PRECISION_BITS 22

/* Prefix of Pillow 10.4.0 ImagingMemoryInstance, from upstream/Imaging.h.
 * Python wrapper enables this ABI only for the pinned Pillow 10.4.0 version.
 */
typedef struct {
    char mode[7];
    int type, depth, bands, xsize, ysize;
    void *palette;
    uint8_t **image8;
} ImagingPrefix;

typedef struct {
    int ksize;
    int *bounds;
    int32_t *coeffs;
} Coefficients;

/* Seven sleeping workers plus the caller, at most eight active CPU threads.
 * Condition variables avoid an OpenMP spin team competing with JPEG decoding
 * or GPU command submission between images. A single call owns the pool until
 * both image passes finish; no work from another image can overlap it. */
static pthread_mutex_t call_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t job_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t job_done = PTHREAD_COND_INITIALIZER;
static pthread_t workers[7];
static int worker_ids[7], worker_count = 0, pool_started = 0, stopping = 0;
static unsigned long generation = 0;
static int job_rows, job_threads, remaining;
static void (*job_function)(void *, int, int);
static void *job_context;

static void *worker_main(void *argument) {
    int id = *(int *)argument;
    unsigned long seen = 0;
    pthread_mutex_lock(&pool_lock);
    for (;;) {
        while (!stopping && generation == seen) pthread_cond_wait(&job_ready, &pool_lock);
        if (stopping) break;
        seen = generation;
        if (id >= job_threads) continue;
        int first = job_rows * id / job_threads;
        int last = job_rows * (id + 1) / job_threads;
        void (*function)(void *, int, int) = job_function;
        void *context = job_context;
        pthread_mutex_unlock(&pool_lock);
        function(context, first, last);
        pthread_mutex_lock(&pool_lock);
        if (--remaining == 0) pthread_cond_signal(&job_done);
    }
    pthread_mutex_unlock(&pool_lock);
    return NULL;
}

static void pool_start(void) {
    if (pool_started) return;
    pool_started = 1;
    for (int i = 0; i < 7; i++) {
        worker_ids[i] = i + 1;
        if (pthread_create(&workers[i], NULL, worker_main, &worker_ids[i])) break;
        worker_count++;
    }
}

static void run_rows(void (*function)(void *, int, int), void *context, int rows, int threads) {
    if (threads > worker_count + 1) threads = worker_count + 1;
    if (threads <= 1) { function(context, 0, rows); return; }
    pthread_mutex_lock(&pool_lock);
    job_rows = rows; job_threads = threads; remaining = threads - 1;
    job_function = function; job_context = context;
    generation++;
    pthread_cond_broadcast(&job_ready);
    pthread_mutex_unlock(&pool_lock);
    function(context, 0, rows / threads);
    pthread_mutex_lock(&pool_lock);
    while (remaining) pthread_cond_wait(&job_done, &pool_lock);
    pthread_mutex_unlock(&pool_lock);
}

static inline double sinc_filter(double x) {
    if (x == 0.0) return 1.0;
    x = x * M_PI;
    return sin(x) / x;
}

static inline double lanczos_filter(double x) {
    if (-3.0 <= x && x < 3.0) return sinc_filter(x) * sinc_filter(x / 3);
    return 0.0;
}

static int coefficients(int input_size, int output_size, Coefficients *result) {
    double scale, filterscale, support, center, ww, ss;
    double *weights;
    int ksize, xx, x, xmin, xmax;
    /* Pillow receives full-box boundaries as float, including integer size. */
    float in0 = 0.0f, in1 = (float)input_size;
    filterscale = scale = (double)(in1 - in0) / output_size;
    if (filterscale < 1.0) filterscale = 1.0;
    support = 3.0 * filterscale;
    if (support > (INT_MAX - 1) / 2) return 0;
    ksize = (int)ceil(support) * 2 + 1;
    if (output_size > INT_MAX / (ksize * (int)sizeof(double))) return 0;
    result->ksize = ksize;
    result->bounds = malloc((size_t)output_size * 2 * sizeof(int));
    result->coeffs = malloc((size_t)output_size * ksize * sizeof(int32_t));
    weights = malloc((size_t)ksize * sizeof(double));
    if (!result->bounds || !result->coeffs || !weights) {
        free(result->bounds); free(result->coeffs); free(weights);
        result->bounds = NULL; result->coeffs = NULL;
        return 0;
    }
    for (xx = 0; xx < output_size; xx++) {
        center = in0 + (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        xmax = (int)(center + support + 0.5);
        if (xmax > input_size) xmax = input_size;
        xmax -= xmin;
        for (x = 0; x < xmax; x++) {
            double w = lanczos_filter((x + xmin - center + 0.5) * ss);
            weights[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0) weights[x] /= ww;
        }
        for (; x < ksize; x++) weights[x] = 0;
        result->bounds[xx * 2] = xmin;
        result->bounds[xx * 2 + 1] = xmax;
        for (x = 0; x < ksize; x++) {
            double w = weights[x];
            result->coeffs[xx * ksize + x] = (int32_t)(
                w < 0 ? -0.5 + w * (1 << PRECISION_BITS)
                      :  0.5 + w * (1 << PRECISION_BITS));
        }
    }
    free(weights);
    return 1;
}

static inline uint8_t clip8(int value) {
    int rounded = value >> PRECISION_BITS;
    return rounded < 0 ? 0 : (rounded > 255 ? 255 : (uint8_t)rounded);
}

static void horizontal(uint8_t *out, const uint8_t *in, int width, const Coefficients *c) {
    for (int xx = 0; xx < width; xx++) {
        int xmin = c->bounds[xx * 2], count = c->bounds[xx * 2 + 1];
        const int32_t *k = c->coeffs + xx * c->ksize;
        int ss0 = 1 << (PRECISION_BITS - 1);
        for (int x = 0; x < count; x++) ss0 += in[x + xmin] * k[x];
        out[xx] = clip8(ss0);
    }
}

static void vertical(uint8_t *out, uint8_t **rows, int width, int yy, const Coefficients *c) {
    int ymin = c->bounds[yy * 2], count = c->bounds[yy * 2 + 1];
    const int32_t *k = c->coeffs + yy * c->ksize;
    for (int xx = 0; xx < width; xx++) {
        int ss0 = 1 << (PRECISION_BITS - 1);
        for (int y = 0; y < count; y++) ss0 += rows[y + ymin][xx] * k[y];
        out[xx] = clip8(ss0);
    }
}

typedef struct {
    uint8_t **source, **target;
    int width;
    const Coefficients *coeffs;
} Pass;

static void horizontal_rows(void *context, int first, int last) {
    Pass *p = context;
    for (int yy = first; yy < last; yy++)
        horizontal(p->target[yy], p->source[yy], p->width, p->coeffs);
}
static void vertical_rows(void *context, int first, int last) {
    Pass *p = context;
    for (int yy = first; yy < last; yy++)
        vertical(p->target[yy], p->source, p->width, yy, p->coeffs);
}
static void copy_rows(void *context, int first, int last) {
    Pass *p = context;
    for (int yy = first; yy < last; yy++)
        memcpy(p->target[yy], p->source[yy], (size_t)p->width);
}

static PyObject *resize_l(PyObject *self, PyObject *args) {
    PyObject *source_capsule, *target_capsule;
    int threads;
    if (!PyArg_ParseTuple(args, "OOi", &source_capsule, &target_capsule, &threads)) return NULL;
    if (threads < 1 || threads > 8) {
        PyErr_SetString(PyExc_ValueError, "Native resampling threads must be in [1, 8]");
        return NULL;
    }
    ImagingPrefix *source = PyCapsule_GetPointer(source_capsule, "PIL Imaging");
    if (!source) return NULL;
    ImagingPrefix *target = PyCapsule_GetPointer(target_capsule, "PIL Imaging");
    if (!target) return NULL;
    if (strcmp(source->mode, "L") || strcmp(target->mode, "L") || source->type != 0 ||
        target->type != 0 || source->bands != 1 || target->bands != 1 ||
        !source->image8 || !target->image8 || source->xsize <= 0 || source->ysize <= 0 ||
        target->xsize <= 0 || target->ysize <= 0 || source == target) {
        PyErr_SetString(PyExc_ValueError, "Expected distinct, positive-size Pillow 10.4 L images");
        return NULL;
    }
    Coefficients h = {0}, v = {0};
    const int do_h = source->xsize != target->xsize;
    const int do_v = source->ysize != target->ysize;
    if ((do_h && !coefficients(source->xsize, target->xsize, &h)) ||
        (do_v && !coefficients(source->ysize, target->ysize, &v))) {
        free(h.bounds); free(h.coeffs); free(v.bounds); free(v.coeffs);
        return PyErr_NoMemory();
    }
    uint8_t *temp = NULL;
    uint8_t **temp_rows = NULL;
    if (do_h && do_v) {
        if ((size_t)source->ysize > SIZE_MAX / (size_t)target->xsize) {
            free(h.bounds); free(h.coeffs); free(v.bounds); free(v.coeffs);
            return PyErr_NoMemory();
        }
        temp = malloc((size_t)source->ysize * target->xsize);
        temp_rows = malloc((size_t)source->ysize * sizeof(uint8_t *));
        if (!temp || !temp_rows) {
            free(temp); free(temp_rows); free(h.bounds); free(h.coeffs); free(v.bounds); free(v.coeffs);
            return PyErr_NoMemory();
        }
        for (int y = 0; y < source->ysize; y++) temp_rows[y] = temp + (size_t)y * target->xsize;
    }
    /* GIL released while waiting/working; source/target owners remain live. */
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&call_lock);
    pool_start();
    Pass pass = {source->image8, do_v && do_h ? temp_rows : target->image8, target->xsize, &h};
    if (do_h) run_rows(horizontal_rows, &pass, source->ysize, threads);
    /* run_rows completes all H workers before the uint8 intermediate feeds V. */
    if (do_v) {
        pass.source = do_h ? temp_rows : source->image8;
        pass.target = target->image8; pass.coeffs = &v;
        run_rows(vertical_rows, &pass, target->ysize, threads);
    }
    if (!do_h && !do_v) run_rows(copy_rows, &pass, target->ysize, threads);
    pthread_mutex_unlock(&call_lock);
    Py_END_ALLOW_THREADS
    free(temp); free(temp_rows); free(h.bounds); free(h.coeffs); free(v.bounds); free(v.coeffs);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"resize_l", resize_l, METH_VARARGS, "Resize distinct full-box Pillow 10.4 L images exactly."},
    {NULL, NULL, 0, NULL}
};
static void shutdown_pool(void *module) {
    if (!pool_started) return;
    pthread_mutex_lock(&pool_lock);
    stopping = 1;
    pthread_cond_broadcast(&job_ready);
    pthread_mutex_unlock(&pool_lock);
    for (int i = 0; i < worker_count; i++) pthread_join(workers[i], NULL);
}
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_core", NULL, -1, methods, NULL, NULL, NULL, shutdown_pool
};
PyMODINIT_FUNC PyInit__core(void) { return PyModule_Create(&module); }
