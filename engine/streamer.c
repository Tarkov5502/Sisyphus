/*
 * engine/streamer.c — Sisyphus build step 2: the tape streamer, alone.
 *
 * Reads the model the way a sweep does — at full drive speed, one io_uring per physical drive,
 * O_DIRECT (no page cache), large aligned reads at a fixed queue depth, into a ring of aligned, pre-faulted
 * buffers (mlock'd when RLIMIT_MEMLOCK allows) that a consumer drains in arrival order. Nothing is interpreted; the consumer touches
 * the bytes (or checksums them with --verify) and hands the slot back. The output is the number
 * the design rests on: sustained GB/s per drive and aggregate, and the implied step time.
 *
 * Two modes:
 *
 *   engine/streamer [opts] FILE...                 file mode: read whole files end to end
 *   engine/streamer [opts] --sweep SHARD_DIR       sweep mode: parse the GGUF shards and read the
 *                                                  model in SWEEP ORDER — block by block, trunk
 *                                                  tensors then expert slabs — exactly the I/O a
 *                                                  step performs. --experts N reads a random N of
 *                                                  the 896 experts per block (a batch's routing
 *                                                  union), which measures the real cost of the
 *                                                  sparse regime: 3.2 MB slab reads instead of
 *                                                  8 MB runs.
 * Options:
 *   --seconds N    wrap around for N seconds (0 = one pass)            --qd 32     queue depth per drive
 *   --bs 8         max request size in MiB                            --verify    full checksum of every byte
 *   --experts N    sweep mode: experts read per block (default all)   --seed S    routing sample seed
 *   --json out     write the result as JSON
 *
 *   make -C engine        (needs liburing-dev)
 *
 * Why this and not fio: fio measures a drive; this measures the engine's I/O path — many files,
 * several drives, sweep order, slab-sized sparse reads, one consumer, arrival-order handoff, with
 * the buffer discipline (pinned, 4 KiB-aligned, reused) the CUDA H2D copies will use. When this
 * hits the fio number, step 2 is done and step 3 consumes the ring for real.
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <liburing.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#define MAX_DEVS 8
#define MAX_FILES 256
#define ALIGN 4096

/* ------------------------------------------------------------------------------------------ */
/*  requests: the unit of I/O. Every request is 4 KiB-aligned (O_DIRECT) and <= bs bytes.      */
/* ------------------------------------------------------------------------------------------ */
typedef struct { int fd; off_t off; uint32_t len; } req_t;
typedef struct { req_t *v; size_t n, cap; } reqlist_t;

static void req_push(reqlist_t *l, int fd, off_t off, uint32_t len) {
    if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 4096; l->v = realloc(l->v, l->cap * sizeof *l->v); if (!l->v) { perror("realloc"); exit(1); } }
    l->v[l->n++] = (req_t){ fd, off, len };
}

/* Byte ranges are accumulated per device, merged when adjacent on the same fd (dense sweeps
 * become long runs), aligned outward to 4 KiB, and split into <= bs requests. */
typedef struct { int fd; off_t start, end, fsize; int valid; } pending_t;

static void flush_pending(reqlist_t *l, pending_t *p, size_t bs) {
    if (!p->valid) return;
    off_t s = p->start / ALIGN * ALIGN;
    off_t e = (p->end + ALIGN - 1) / ALIGN * ALIGN;
    off_t fe = (p->fsize + ALIGN - 1) / ALIGN * ALIGN;
    if (e > fe) e = fe;                                   /* the kernel short-reads at EOF */
    for (off_t o = s; o < e; o += (off_t)bs) { off_t n = e - o; if (n > (off_t)bs) n = (off_t)bs; req_push(l, p->fd, o, (uint32_t)n); }
    p->valid = 0;
}

static void add_range(reqlist_t *l, pending_t *p, int fd, off_t off, off_t len, size_t bs, off_t fsize) {
    if (p->valid && p->fd == fd && off >= p->start && off <= p->end + ALIGN) { if (off + len > p->end) p->end = off + len; return; }
    flush_pending(l, p, bs);
    *p = (pending_t){ fd, off, off + len, fsize, 1 };
}

/* ------------------------------------------------------------------------------------------ */
/*  files and devices                                                                           */
/* ------------------------------------------------------------------------------------------ */
typedef struct { char *path; int fd; off_t size; dev_t dev; int devidx; } file_t;
static file_t g_files[MAX_FILES]; static int g_nfiles = 0;

static int dev_index(dev_t dev);
static file_t *open_file(const char *path) {
    if (g_nfiles == MAX_FILES) { fprintf(stderr, "too many files\n"); exit(1); }
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(1); }
    struct stat st; if (fstat(fd, &st) != 0) { perror("fstat"); exit(1); }
    file_t *f = &g_files[g_nfiles++];
    f->path = strdup(path); f->fd = fd; f->size = st.st_size; f->dev = st.st_dev;
    f->devidx = dev_index(st.st_dev);
    return f;
}
static file_t *file_by_fd(int fd) { for (int i = 0; i < g_nfiles; i++) if (g_files[i].fd == fd) return &g_files[i]; return NULL; }

typedef struct devstate_s {
    dev_t dev; char name[64];
    reqlist_t reqs;               /* this device's requests, in sweep order */
    int qd; size_t bs; int seconds; int verify;
    atomic_uint_fast64_t bytes;
    uint64_t total_bytes, reads, checksum, short_reads; double elapsed_s;
    int err; char errmsg[160]; int mlock_failed;
} devstate_t;
static devstate_t g_devs[MAX_DEVS]; static int g_ndev = 0;

/* /dev/nvmeXnY for a st_dev via /sys/dev/block/MAJ:MIN -> ../../nvmeXnYpZ (partition stripped) */
static void dev_name(dev_t dev, char *out, size_t n) {
    char link[256], path[64];
    snprintf(path, sizeof path, "/sys/dev/block/%u:%u", major(dev), minor(dev));
    ssize_t k = readlink(path, link, sizeof link - 1);
    if (k > 0) {
        link[k] = 0; const char *base = strrchr(link, '/'); base = base ? base + 1 : link;
        snprintf(out, n, "%.63s", base);
        if (strncmp(out, "nvme", 4) == 0) { char *p = strchr(out + 4, 'p'); if (p && isdigit((unsigned char)p[1])) *p = 0; }
        return;
    }
    snprintf(out, n, "%u:%u", major(dev), minor(dev));
}

/* Devices are keyed by the PARENT block device name (nvme0n1), so two partitions of one NVMe share
 * one ring and are not counted twice. */
static devstate_t *dev_for(dev_t dev) {
    char name[64]; dev_name(dev, name, sizeof name);
    for (int k = 0; k < g_ndev; k++) if (!strcmp(g_devs[k].name, name)) return &g_devs[k];
    if (g_ndev == MAX_DEVS) { fprintf(stderr, "too many devices\n"); exit(1); }
    devstate_t *d = &g_devs[g_ndev++]; memset(d, 0, sizeof *d); d->dev = dev; snprintf(d->name, sizeof d->name, "%s", name);
    return d;
}
static int dev_index(dev_t dev) { return (int)(dev_for(dev) - g_devs); }
static int dev_index_for_fd(int fd) { return file_by_fd(fd)->devidx; }

/* ------------------------------------------------------------------------------------------ */
/*  GGUF index (sweep mode)                                                                     */
/* ------------------------------------------------------------------------------------------ */
typedef struct {
    char name[128]; uint64_t dims[4]; int ndims; uint32_t type;
    off_t off, size;               /* absolute byte range within the shard */
    int fd; int block; int kind;   /* kind: 0 global, 1 trunk, 2 experts */
} tensor_t;
static tensor_t *g_tens = NULL; static size_t g_ntens = 0, g_tcap = 0; static int g_expert_blocks = 0;

static uint64_t rd_u(FILE *f, int bytes) { uint64_t v = 0; if (fread(&v, bytes, 1, f) != 1) { fprintf(stderr, "gguf: short read\n"); exit(1); } return v; }
static void rd_str(FILE *f, char *out, size_t cap) {
    uint64_t n = rd_u(f, 8); if (n > (1u << 20)) { fprintf(stderr, "gguf: absurd string length\n"); exit(1); } char *tmp = malloc(n + 1); if (!tmp) { perror("malloc"); exit(1); }
    if (fread(tmp, 1, n, f) != n) { fprintf(stderr, "gguf: short string\n"); exit(1); }
    tmp[n] = 0; if (out) snprintf(out, cap, "%s", tmp); free(tmp);
}
static int gguf_type_size(uint32_t t) { switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4; case 10: case 11: case 12: return 8; default: return -1; } }
static void skip_value(FILE *f, uint32_t t, uint32_t *align_out) {
    if (t == 8) { rd_str(f, NULL, 0); return; }
    if (t == 9) { uint32_t et = (uint32_t)rd_u(f, 4); uint64_t n = rd_u(f, 8); for (uint64_t i = 0; i < n; i++) skip_value(f, et, NULL); return; }
    int sz = gguf_type_size(t); if (sz < 0) { fprintf(stderr, "gguf: unknown kv type %u\n", t); exit(1); }
    uint64_t v = rd_u(f, sz); if (align_out) *align_out = (uint32_t)v;
}
static int cmp_tensor_off(const void *a, const void *b) { const tensor_t *x = a, *y = b; return (x->off > y->off) - (x->off < y->off); }
static int cmp_int_asc(const void *a, const void *b) { return (*(const int *)a > *(const int *)b) - (*(const int *)a < *(const int *)b); }
static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static void index_shard(file_t *shard) {
    FILE *f = fopen(shard->path, "rb"); if (!f) { perror(shard->path); exit(1); }
    char magic[4]; if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4)) { fprintf(stderr, "%s: not a GGUF\n", shard->path); exit(1); }
    uint32_t version = (uint32_t)rd_u(f, 4);
    if (version < 2 || version > 3) { fprintf(stderr, "%s: GGUF version %u not supported (2 or 3)\n", shard->path, version); exit(1); }
    uint64_t nt = rd_u(f, 8), nkv = rd_u(f, 8); uint32_t align = 32;
    for (uint64_t i = 0; i < nkv; i++) {
        char key[256]; rd_str(f, key, sizeof key); uint32_t t = (uint32_t)rd_u(f, 4);
        uint32_t a = 0; skip_value(f, t, &a); if (!strcmp(key, "general.alignment")) align = a;
    }
    if (align == 0 || (align & (align - 1))) { fprintf(stderr, "%s: bad general.alignment %u\n", shard->path, align); exit(1); }
    size_t first = g_ntens;
    for (uint64_t i = 0; i < nt; i++) {
        if (g_ntens == g_tcap) { g_tcap = g_tcap ? g_tcap * 2 : 4096; g_tens = realloc(g_tens, g_tcap * sizeof *g_tens); if (!g_tens) { perror("realloc"); exit(1); } }
        tensor_t *T = &g_tens[g_ntens++]; memset(T, 0, sizeof *T);
        rd_str(f, T->name, sizeof T->name); T->ndims = (int)rd_u(f, 4);
        if (T->ndims < 1 || T->ndims > 4) { fprintf(stderr, "%s: tensor %s has %d dims\n", shard->path, T->name, T->ndims); exit(1); }
        for (int d = 0; d < T->ndims; d++) T->dims[d] = rd_u(f, 8);
        T->type = (uint32_t)rd_u(f, 4); uint64_t o = rd_u(f, 8); T->fd = shard->fd;
        if (o > (uint64_t)shard->size) { fprintf(stderr, "%s: tensor %s offset %" PRIu64 " beyond file\n", shard->path, T->name, o); exit(1); }
        T->off = (off_t)o;
    }
    off_t hdr_end = ftello(f); fclose(f);
    off_t data = (hdr_end + align - 1) / align * align;
    qsort(g_tens + first, g_ntens - first, sizeof *g_tens, cmp_tensor_off);
    for (size_t i = first; i < g_ntens; i++) {
        off_t end = (i + 1 < g_ntens) ? g_tens[i + 1].off : shard->size - data;
        tensor_t *T = &g_tens[i];
        if (end < T->off) { fprintf(stderr, "%s: tensor %s has negative size (corrupt offsets)\n", shard->path, T->name); exit(1); }
        T->size = end - T->off; T->off += data; T->block = -1; T->kind = 0;
        if (!strncmp(T->name, "blk.", 4)) {
            char *endp; long b = strtol(T->name + 4, &endp, 10);
            if (endp == T->name + 4 || *endp != '.' || b < 0 || b > 4096) { fprintf(stderr, "%s: unexpected tensor name %s\n", shard->path, T->name); exit(1); }
            T->block = (int)b; T->kind = strstr(T->name, "_exps") ? 2 : 1;
            if (T->kind == 2 && T->ndims < 2) { fprintf(stderr, "%s: expert tensor %s has %d dims (expected >= 2, expert index last)\n", shard->path, T->name, T->ndims); exit(1); }
        }
    }
}

/* xorshift for the routing sample */
static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static uint64_t rng(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }

/* Build the sweep-order request lists. experts_per_block <= 0 means all. Returns bytes planned. */
static uint64_t build_sweep(const char *dir, int *experts_per_block, size_t bs, uint64_t *trunk_bytes, uint64_t *expert_bytes, int *nblocks_out, int *nexp_out) {
    DIR *dp = opendir(dir); if (!dp) { perror(dir); exit(1); }
    char *names[MAX_FILES]; int n = 0; struct dirent *de;
    while ((de = readdir(dp))) { size_t L = strlen(de->d_name); if (L > 5 && !strcmp(de->d_name + L - 5, ".gguf") && n < MAX_FILES) names[n++] = strdup(de->d_name); }
    closedir(dp);
    if (!n) { fprintf(stderr, "no .gguf in %s\n", dir); exit(1); }
    qsort(names, n, sizeof *names, cmp_str);
    for (int i = 0; i < n; i++) { char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, names[i]); file_t *sh = open_file(path); index_shard(sh); free(names[i]); }

    int maxblk = -1; for (size_t i = 0; i < g_ntens; i++) if (g_tens[i].block > maxblk) maxblk = g_tens[i].block;
    int nexp = 0, have_exps = 0;
    for (size_t i = 0; i < g_ntens; i++) if (g_tens[i].kind == 2) {                       /* slowest dim = expert */
        uint64_t ne = g_tens[i].dims[g_tens[i].ndims - 1]; have_exps = 1;
        if (ne < 1 || ne > 65536) { fprintf(stderr, "%s: n_expert %" PRIu64 " out of range\n", g_tens[i].name, ne); exit(1); }
        if (!nexp) nexp = (int)ne;
        else if ((int)ne != nexp) { fprintf(stderr, "%s: n_expert %" PRIu64 " differs from %d\n", g_tens[i].name, ne, nexp); exit(1); }
    }
    if (!have_exps && *experts_per_block > 0) fprintf(stderr, "  note: no expert tensors in this model; --experts ignored\n");
    int expert_blocks = 0; for (int b = 0; b <= maxblk; b++) { for (size_t i = 0; i < g_ntens; i++) if (g_tens[i].block == b && g_tens[i].kind == 2) { expert_blocks++; break; } }
    g_expert_blocks = expert_blocks;
    if (*experts_per_block <= 0 || *experts_per_block > nexp) *experts_per_block = nexp;
    int epb = *experts_per_block;
    *nblocks_out = maxblk + 1; *nexp_out = nexp; *trunk_bytes = *expert_bytes = 0;

    pending_t pend[MAX_DEVS]; memset(pend, 0, sizeof pend);
    for (size_t i = 0; i < g_ntens; i++) if (g_tens[i].kind == 0) {          /* globals (embeddings, output head) */
        tensor_t *T = &g_tens[i]; int k = dev_index_for_fd(T->fd);
        add_range(&g_devs[k].reqs, &pend[k], T->fd, T->off, T->size, bs, file_by_fd(T->fd)->size); *trunk_bytes += T->size;
    }
    int *pick = malloc(sizeof(int) * (nexp ? nexp : 1));
    for (int blk = 0; blk <= maxblk; blk++) {
        for (size_t i = 0; i < g_ntens; i++) if (g_tens[i].block == blk && g_tens[i].kind == 1) {      /* trunk of this block */
            tensor_t *T = &g_tens[i]; int k = dev_index_for_fd(T->fd);
            add_range(&g_devs[k].reqs, &pend[k], T->fd, T->off, T->size, bs, file_by_fd(T->fd)->size); *trunk_bytes += T->size;
        }
        if (nexp) {  /* this block's routing union: epb distinct experts, ascending (partial Fisher-Yates) */
            for (int e = 0; e < nexp; e++) pick[e] = e;
            for (int e = 0; e < epb; e++) { int j = e + (int)(rng() % (uint64_t)(nexp - e)); int t = pick[e]; pick[e] = pick[j]; pick[j] = t; }
            qsort(pick, epb, sizeof(int), cmp_int_asc);
        }
        for (size_t i = 0; i < g_ntens; i++) if (g_tens[i].block == blk && g_tens[i].kind == 2) {      /* expert slabs: gate, up, down */
            tensor_t *T = &g_tens[i]; int k = dev_index_for_fd(T->fd); off_t fs = file_by_fd(T->fd)->size;
            /* T->size is the offset delta = true bytes + alignment pad (< align). The per-expert slab is exact
             * only if the pad is smaller than n_expert; otherwise refuse rather than drift into padding. */
            if (T->size % nexp >= 4096) { fprintf(stderr, "%s: size %" PRId64 " not a clean multiple of %d experts (alignment too large for slab arithmetic)\n", T->name, (int64_t)T->size, nexp); exit(1); }
            /* pad < 4096 but >= nexp: slab drifts by < 4 KiB over all experts, absorbed by the 4 KiB outward alignment */
            off_t slab = T->size / nexp;
            if (epb == nexp) { add_range(&g_devs[k].reqs, &pend[k], T->fd, T->off, T->size, bs, fs); *expert_bytes += T->size; }
            else for (int e = 0; e < epb; e++) { add_range(&g_devs[k].reqs, &pend[k], T->fd, T->off + slab * pick[e], slab, bs, fs); *expert_bytes += slab; }
        }
    }
    free(pick);
    for (int k = 0; k < g_ndev; k++) flush_pending(&g_devs[k].reqs, &pend[k], bs);
    return *trunk_bytes + *expert_bytes;
}

/* ------------------------------------------------------------------------------------------ */
/*  the device thread: drive its request list through io_uring at queue depth qd               */
/* ------------------------------------------------------------------------------------------ */
typedef struct { void *buf; size_t idx; } slot_t;
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static uint64_t consume(const uint8_t *p, size_t n, int verify) {
    uint64_t h = 1469598103934665603ULL;
    if (verify) {   /* four independent FNV lanes so the hash is not one serial multiply chain (~4x faster) */
        const uint64_t *w = (const uint64_t *)p; size_t nw = n / 8, i = 0;
        uint64_t a = h, b = h ^ 0x9E3779B97F4A7C15ULL, c = h ^ 0xC2B2AE3D27D4EB4FULL, e = h ^ 0x165667B19E3779F9ULL;
        for (; i + 4 <= nw; i += 4) { a ^= w[i]; a *= 1099511628211ULL; b ^= w[i+1]; b *= 1099511628211ULL; c ^= w[i+2]; c *= 1099511628211ULL; e ^= w[i+3]; e *= 1099511628211ULL; }
        for (; i < nw; i++) { a ^= w[i]; a *= 1099511628211ULL; }
        h = a ^ (b * 31) ^ (c * 131) ^ (e * 1031);
    }
    else for (size_t i = 0; i < n; i += ALIGN) h += p[i];
    return h;
}

static void *dev_thread(void *arg) {
    devstate_t *d = arg;
    if (d->reqs.n == 0) return NULL;
    struct io_uring ring;
    int rc = io_uring_queue_init(d->qd * 2, &ring, 0);
    if (rc < 0) { d->err = -rc; snprintf(d->errmsg, sizeof d->errmsg, "io_uring_queue_init: %s", strerror(-rc)); return NULL; }
    slot_t *slots = calloc(d->qd, sizeof *slots); int *fs = malloc(sizeof(int) * d->qd), nfree = d->qd;
    int *retry = calloc(d->qd, sizeof(int));
    if (!slots || !fs || !retry) { d->err = ENOMEM; snprintf(d->errmsg, sizeof d->errmsg, "out of memory"); io_uring_queue_exit(&ring); return NULL; }
    for (int i = 0; i < d->qd; i++) {
        if (posix_memalign(&slots[i].buf, ALIGN, d->bs) != 0) {
            d->err = ENOMEM; snprintf(d->errmsg, sizeof d->errmsg, "posix_memalign(%zu MiB x %d) failed", d->bs >> 20, d->qd);
            for (int j = 0; j < i; j++) free(slots[j].buf);
            free(slots); free(fs); free(retry); io_uring_queue_exit(&ring); return NULL;
        }
        memset(slots[i].buf, 0, d->bs);
        if (mlock(slots[i].buf, d->bs) != 0 && i == 0) d->mlock_failed = 1;   /* RLIMIT_MEMLOCK is usually 8 MiB; O_DIRECT pins pages per DMA anyway */
        fs[i] = i;
    }
    size_t next = 0; int inflight = 0, stop = 0;
    double t0 = now_s(), tend = d->seconds > 0 ? t0 + d->seconds : 0;   /* after buffer allocation, so setup is not timed */
    uint64_t total = 0, reads = 0, sum = 0, short_reads = 0;
    /* a short read that is not at EOF is re-issued for the remainder; slots carry the live request */
    req_t *live = calloc(d->qd, sizeof *live);
    for (;;) {
        while (nfree > 0 && !stop && !d->err) {
            if (next >= d->reqs.n) { if (d->seconds > 0 && now_s() < tend) next = 0; else { stop = 1; break; } }
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring); if (!sqe) break;
            int s = fs[--nfree]; slots[s].idx = next; live[s] = d->reqs.v[next++]; retry[s] = 0;
            io_uring_prep_read(sqe, live[s].fd, slots[s].buf, live[s].len, live[s].off);
            io_uring_sqe_set_data(sqe, (void *)(intptr_t)s); inflight++;
        }
        if (inflight > 0) { rc = io_uring_submit(&ring); if (rc < 0) { d->err = -rc; snprintf(d->errmsg, sizeof d->errmsg, "io_uring_submit: %s", strerror(-rc)); break; } }
        if (inflight == 0) break;
        if (d->seconds > 0 && now_s() >= tend) stop = 1;
        struct io_uring_cqe *cqe;
        rc = io_uring_wait_cqe(&ring, &cqe);
        if (rc < 0) { d->err = -rc; snprintf(d->errmsg, sizeof d->errmsg, "io_uring_wait_cqe: %s", strerror(-rc)); break; }
        for (;;) {
            int s = (int)(intptr_t)io_uring_cqe_get_data(cqe); int res = cqe->res; io_uring_cqe_seen(&ring, cqe); inflight--;
            req_t *r = &live[s];
            if (res < 0 && !d->err) {
                file_t *f = file_by_fd(r->fd);
                d->err = -res; snprintf(d->errmsg, sizeof d->errmsg, "read %s @%" PRId64 " len %u: %s%s", f ? f->path : "?", (int64_t)r->off, r->len,
                                        strerror(-res), res == -EINVAL ? " (O_DIRECT refused: alignment/filesystem)" : "");
                stop = 1;                           /* keep reaping: the other in-flight reads still own their buffers */
            } else if (res >= 0 && !d->err) {
                if (res > 0) { sum ^= consume(slots[s].buf, (size_t)res, d->verify); total += (uint64_t)res; reads++; atomic_store(&d->bytes, total); }
                file_t *f = file_by_fd(r->fd);
                if ((uint32_t)res < r->len && f && r->off + res < f->size) {
                    /* short (or empty) read mid-file (O_DIRECT punted to io-wq can do this): re-issue the remainder, 4 KiB aligned */
                    short_reads++;
                    off_t noff = (r->off + res) / ALIGN * ALIGN; uint32_t nlen = (uint32_t)(r->off + r->len - noff);
                    if (noff == r->off && ++retry[s] > 8) { d->err = EIO; snprintf(d->errmsg, sizeof d->errmsg, "read %s @%" PRId64 ": repeated short reads (%d bytes)", f->path, (int64_t)r->off, res); stop = 1; }
                    else {
                        total -= (uint64_t)((r->off + res) - noff);          /* the realigned bytes will be counted again */
                        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                        if (sqe) {
                            live[s] = (req_t){ r->fd, noff, nlen }; io_uring_prep_read(sqe, r->fd, slots[s].buf, nlen, noff); io_uring_sqe_set_data(sqe, (void *)(intptr_t)s); inflight++;
                            rc = io_uring_submit(&ring); if (rc < 0) { d->err = -rc; snprintf(d->errmsg, sizeof d->errmsg, "io_uring_submit (requeue): %s", strerror(-rc)); stop = 1; }
                            goto next_cqe;
                        }
                        d->err = EIO; snprintf(d->errmsg, sizeof d->errmsg, "requeue: submission queue full (should not happen)"); stop = 1;
                    }
                }
            }
            fs[nfree++] = s;
        next_cqe:
            if (io_uring_peek_cqe(&ring, &cqe) != 0) break;
        }
        if (d->err && inflight == 0) break;
    }
    d->elapsed_s = now_s() - t0; d->total_bytes = total; d->reads = reads; d->checksum = sum; d->short_reads = short_reads;
    free(live); free(retry);
    for (int i = 0; i < d->qd; i++) { munlock(slots[i].buf, d->bs); free(slots[i].buf); }
    free(slots); free(fs); io_uring_queue_exit(&ring);
    return NULL;
}

static void json_escape(const char *in, char *out, size_t cap) {
    size_t o = 0;
    for (; *in && o + 6 < cap; in++) {
        unsigned char c = (unsigned char)*in;
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = (char)c; }
        else if (c < 0x20) { o += (size_t)snprintf(out + o, cap - o, "\\u%04x", c); }
        else out[o++] = (char)c;
    }
    out[o] = 0;
}

/* ------------------------------------------------------------------------------------------ */
static void usage(void) {
    fprintf(stderr, "usage: streamer [--seconds N] [--qd 32] [--bs 8] [--verify] [--json out] [--seed S] (FILE... | --sweep DIR [--experts N])\n"
                    "       options must come before the file list\n");
    exit(2);
}

int main(int argc, char **argv) {
    int qd = 32, seconds = 0, verify = 0, experts = 0; size_t bs_mb = 8; const char *json = NULL, *sweep = NULL;
    int i = 1;
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--qd") && i + 1 < argc) qd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bs") && i + 1 < argc) bs_mb = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) g_rng ^= (uint64_t)strtoull(argv[++i], NULL, 10) * 0x2545F4914F6CDD1DULL;
        else if (!strcmp(argv[i], "--verify")) verify = 1;
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
        else if (!strcmp(argv[i], "--sweep") && i + 1 < argc) sweep = argv[++i];
        else if (argv[i][0] == '-' && argv[i][1]) usage();
        else break;
    }
    if (qd < 1 || qd > 256 || bs_mb < 1 || bs_mb > 64 || seconds < 0) usage();
    size_t bs = bs_mb << 20;
    uint64_t planned = 0, trunk_b = 0, exp_b = 0; int nblocks = 0, nexp = 0;

    if (sweep) {
        if (i < argc) usage();
        planned = build_sweep(sweep, &experts, bs, &trunk_b, &exp_b, &nblocks, &nexp);
        int sparse = experts < nexp;
        fprintf(stderr, "streamer: SWEEP of %s: %zu tensors in %d shards, %d blocks, %d experts/block%s\n", sweep, g_ntens, g_nfiles, nblocks, nexp, sparse ? " (sampled)" : "");
        fprintf(stderr, "  plan: trunk %.1f GB + experts %.1f GB = %.1f GB per step%s\n", trunk_b / 1e9, exp_b / 1e9, planned / 1e9, sparse ? "" : " (dense: every expert)");
        if (sparse) fprintf(stderr, "  sparse: %d of %d experts per block (%d blocks have experts), slab %.2f MB x 3 tensors\n", experts, nexp, g_expert_blocks, g_expert_blocks && experts ? exp_b / 1e6 / g_expert_blocks / experts / 3 : 0.0);
        if (verify) fprintf(stderr, "  note: --verify hashes every byte in the I/O thread; treat the GB/s as a lower bound\n");
    } else {
        if (i >= argc) usage();
        for (; i < argc; i++) { file_t *f = open_file(argv[i]); devstate_t *d = dev_for(f->dev); pending_t p = {0}; add_range(&d->reqs, &p, f->fd, 0, f->size, bs, f->size); flush_pending(&d->reqs, &p, bs); planned += (uint64_t)f->size; }
        fprintf(stderr, "streamer: %d file(s), %.1f GB\n", g_nfiles, planned / 1e9);
    }
    {   /* the pinned rings must leave room for everything else: refuse more than half of RAM */
        long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
        uint64_t need = (uint64_t)qd * bs * (uint64_t)g_ndev, ram = (pages > 0 && psz > 0) ? (uint64_t)pages * (uint64_t)psz : 0;
        if (ram && need > ram / 2) { fprintf(stderr, "qd %d x bs %zu MiB x %d device(s) = %.1f GiB of pinned buffers exceeds half of RAM (%.1f GiB); lower --qd or --bs\n", qd, bs_mb, g_ndev, need / 1073741824.0, ram / 1073741824.0); return 2; }
    }
    fprintf(stderr, "  bs %zu MiB, qd %d/device, %s, O_DIRECT + io_uring%s\n", bs_mb, qd, seconds ? "timed" : "one pass", verify ? ", full checksum" : "");
    if (!sweep && verify) fprintf(stderr, "  note: --verify hashes every byte in the I/O thread; treat the GB/s as a lower bound\n");
    for (int k = 0; k < g_ndev; k++) {
        devstate_t *d = &g_devs[k]; d->qd = qd; d->bs = bs; d->seconds = seconds; d->verify = verify;
        uint64_t b = 0; for (size_t r = 0; r < d->reqs.n; r++) b += d->reqs.v[r].len;
        fprintf(stderr, "  %-10s %8zu requests %8.1f GB  avg %.2f MiB\n", d->name, d->reqs.n, b / 1e9, d->reqs.n ? b / (double)d->reqs.n / 1048576 : 0.0);
    }

    pthread_t th[MAX_DEVS]; int joined[MAX_DEVS] = {0}; double t0 = now_s();
    for (int k = 0; k < g_ndev; k++) if (pthread_create(&th[k], NULL, dev_thread, &g_devs[k]) != 0) { perror("pthread_create"); return 1; }
    uint64_t last[MAX_DEVS] = {0}; double tl = t0; int alive = 1;
    fprintf(stderr, "%7s", "t(s)"); for (int k = 0; k < g_ndev; k++) fprintf(stderr, " %10s", g_devs[k].name); fprintf(stderr, " %10s %10s\n", "aggregate", "read GB");
    while (alive) {
        struct timespec ts = { 1, 0 }; nanosleep(&ts, NULL);
        double t = now_s(), dt = t - tl; tl = t; double agg = 0; uint64_t cum = 0;
        fprintf(stderr, "%7.0f", t - t0);
        for (int k = 0; k < g_ndev; k++) { uint64_t b = atomic_load(&g_devs[k].bytes); double r = (b - last[k]) / dt / 1e9; last[k] = b; agg += r; cum += b; fprintf(stderr, " %10.2f", r); }
        fprintf(stderr, " %10.2f %10.1f%s", agg, cum / 1e9, isatty(2) ? "\r" : "\n");
        alive = 0; for (int k = 0; k < g_ndev; k++) { if (joined[k]) continue; if (pthread_tryjoin_np(th[k], NULL) == 0) joined[k] = 1; else alive = 1; }
    }
    fprintf(stderr, "\n");
    double wall = now_s() - t0;

    int err = 0; double agg = 0; uint64_t tot = 0; double longest = 0;
    for (int k = 0; k < g_ndev; k++) {
        devstate_t *d = &g_devs[k];
        if (d->err) { fprintf(stderr, "  %-10s ERROR: %s\n", d->name, d->errmsg); err = 1; continue; }
        if (!d->reqs.n) continue;
        double gbps = d->total_bytes / d->elapsed_s / 1e9; agg += gbps; tot += d->total_bytes; if (d->elapsed_s > longest) longest = d->elapsed_s;
        fprintf(stderr, "  %-10s %8.1f GB in %6.1f s  = %5.2f GB/s  (%" PRIu64 " reads, %.2f MiB avg)%s\n", d->name, d->total_bytes / 1e9, d->elapsed_s, gbps, d->reads, d->reads ? d->total_bytes / (double)d->reads / 1048576 : 0.0, "");
        if (verify) fprintf(stderr, "  %-10s checksum %016" PRIx64 "\n", "", d->checksum);
    }
    uint64_t shorts = 0; int mlf = 0; for (int k = 0; k < g_ndev; k++) { shorts += g_devs[k].short_reads; mlf |= g_devs[k].mlock_failed; }
    if (mlf) fprintf(stderr, "  (mlock of the ring buffers refused: RLIMIT_MEMLOCK; harmless for O_DIRECT, pages are pinned per DMA)\n");
    if (shorts) fprintf(stderr, "  (%" PRIu64 " short reads re-issued)\n", shorts);
    if (seconds) fprintf(stderr, "  aggregate %.2f GB/s (drives concurrently busy for %d s)  =>  861 GB dense sweep at this rate = %.0f s\n", agg, seconds, agg > 0 ? 861.0 / agg : 0.0);
    else {
        fprintf(stderr, "  one pass: %.1f GB in %.1f s (slowest drive) = %.2f GB/s effective; drives summed while busy %.2f GB/s\n", tot / 1e9, longest, longest > 0 ? tot / longest / 1e9 : 0.0, agg);
        if (sweep) fprintf(stderr, "  => one STEP of this plan (%.1f GB planned) took %.1f s\n", planned / 1e9, longest);
        if (tot != planned) fprintf(stderr, "  (read %" PRIu64 " bytes vs %" PRIu64 " planned; the difference is 4 KiB alignment at slab and file ends%s)\n", tot, planned,
                                    (tot + 4096 * (uint64_t)g_nfiles < planned) ? " -- NO: something was skipped" : "");
    }
    if (json) {
        FILE *j = fopen(json, "w");
        if (j) {
            double eff = seconds ? agg : (longest > 0 ? tot / longest / 1e9 : 0.0);
            char step_buf[32]; snprintf(step_buf, sizeof step_buf, "%.3f", longest);
            fprintf(j, "{\"mode\":\"%s\",\"experts_per_block\":%d,\"blocks\":%d,\"expert_blocks\":%d,\"planned_bytes\":%" PRIu64 ",\"trunk_bytes\":%" PRIu64 ",\"expert_bytes\":%" PRIu64
                       ",\"bs_mib\":%zu,\"qd\":%d,\"seconds\":%d,\"verify\":%d,\"wall_s\":%.3f,\"step_s\":%s,\"aggregate_gbps\":%.4f,\"effective_gbps\":%.4f,\"short_reads\":%" PRIu64 ",\"devices\":[",
                    sweep ? "sweep" : "files", experts, nblocks, g_expert_blocks, planned, trunk_b, exp_b, bs_mb, qd, seconds, verify, wall, seconds ? "null" : step_buf, agg, eff, shorts);
            for (int k = 0; k < g_ndev; k++) {
                devstate_t *d = &g_devs[k]; uint64_t b = 0; for (size_t r = 0; r < d->reqs.n; r++) b += d->reqs.v[r].len;
                char ename[128], eerr[400]; json_escape(d->name, ename, sizeof ename); json_escape(d->err ? d->errmsg : "", eerr, sizeof eerr);
                fprintf(j, "%s{\"name\":\"%s\",\"requests\":%zu,\"planned_bytes\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"elapsed_s\":%.3f,\"gbps\":%.4f,\"reads\":%" PRIu64 ",\"short_reads\":%" PRIu64 ",\"error\":\"%s\"}",
                        k ? "," : "", ename, d->reqs.n, b, d->total_bytes, d->elapsed_s, d->elapsed_s > 0 ? d->total_bytes / d->elapsed_s / 1e9 : 0.0, d->reads, d->short_reads, eerr);
            }
            fprintf(j, "]}\n"); fclose(j); fprintf(stderr, "  wrote %s\n", json);
        }
    }
    return err;
}
