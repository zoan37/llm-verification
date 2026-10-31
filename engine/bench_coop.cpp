// bench-coop: what does cooperative verification cost a llama.cpp provider?
//
// Generates B sequences in parallel (one batched decode per step, as a serving engine does) and picks each next
// token three ways, timing the whole loop and the sampling part on its own:
//   default   llama.cpp's usual sampler chain (top_k 40, top_p 0.95, min_p 0.05, temperature 0.8)
//   logprobs  the same, plus the top-20 logprobs a serving engine returns when asked (full-vocab log-softmax)
//   coop      the cooperative rule: top-20 logprobs, then seeded Gumbel-max over them (docs/receipt-spec.md),
//             SHA-256 noise per candidate exactly as llmverify/sampling.py
//   coop-fast the same with a one-pass top-20 and a float log-sum-exp
//   coop-mt   coop-fast with the per-request work spread over CPU threads
// Per-sequence sampling runs on one thread (as llama-server does it per slot), except in coop-mt.
//
//   bench-coop -m model.gguf -n 128 -b 1,8,32 [-ngl 99]
#include "llama.h"
extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static double gumbel(uint64_t seed, const std::string & key, int step, int tok) {
    char buf[256];
    int n = snprintf(buf, sizeof(buf), "llmverify-gumbel-v1:%llu:%s:%d:%d", (unsigned long long) seed, key.c_str(), step, tok);
    sha256_t ctx;
    unsigned char h[32];
    sha256_init(&ctx);
    sha256_update(&ctx, (const unsigned char *) buf, n);
    sha256_final(&ctx, h);
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | h[i];
    double u = ((double) v + 0.5) / 18446744073709551616.0;
    return -std::log(-std::log(u));
}

// top-20 (id, logprob) over the full vocabulary: one pass for the max, one for the log-sum-exp, a partial sort.
static void top20(const float * lg, int n_vocab, std::vector<std::pair<int, double>> & out, std::vector<int> & idx) {
    float mx = -INFINITY;
    for (int v = 0; v < n_vocab; v++) mx = std::max(mx, lg[v]);
    double z = 0;
    for (int v = 0; v < n_vocab; v++) z += std::exp((double) lg[v] - mx);
    const double lse = mx + std::log(z);
    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    out.clear();
    for (int i = 0; i < 20; i++) out.emplace_back(idx[i], (double) lg[idx[i]] - lse);
}

// Faster: one pass keeps the max and a sorted top-20 (insertion only when a value beats the 20th), a second sums
// exponentials in float and skips tokens more than 30 nats below the max (their share is < 1e-13, far below the
// 1e-4-nat rounding the receipt commits to).
static void top20_fast(const float * lg, int n_vocab, std::vector<std::pair<int, double>> & out) {
    int ids[20]; float vals[20]; int n = 0;
    float mx = -INFINITY;
    for (int v = 0; v < n_vocab; v++) {
        const float x = lg[v];
        mx = x > mx ? x : mx;
        if (n < 20 || x > vals[n - 1]) {
            int j = n < 20 ? n++ : 19;
            while (j > 0 && vals[j - 1] < x) { vals[j] = vals[j - 1]; ids[j] = ids[j - 1]; j--; }
            vals[j] = x; ids[j] = v;
        }
    }
    float z = 0.f;
    const float cut = mx - 30.f;
    for (int v = 0; v < n_vocab; v++) if (lg[v] > cut) z += expf(lg[v] - mx);
    const double lse = mx + std::log((double) z);
    out.clear();
    for (int i = 0; i < n; i++) out.emplace_back(ids[i], (double) vals[i] - lse);
}

static int coop_choose(const float * lg, int n_vocab, int s, int step, std::vector<std::pair<int, double>> & top) {
    top20_fast(lg, n_vocab, top);
    const double T = 0.8;
    const std::string key = "bench-request-" + std::to_string(s);
    int best = -1; double bv = -INFINITY;
    for (auto & c : top) {
        double v = c.second / T + gumbel(42, key, step, c.first);
        if (v > bv || (v == bv && c.first < best)) { bv = v; best = c.first; }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_gen = 128, ngl = 99;
    std::vector<int> batches = {1, 8, 32};
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-m") model_path = argv[++i];
        else if (a == "-n") n_gen = atoi(argv[++i]);
        else if (a == "-ngl") ngl = atoi(argv[++i]);
        else if (a == "-b") {
            batches.clear();
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) batches.push_back(atoi(t.c_str()));
        }
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const std::string prompt = "Write a short story about a lighthouse keeper who adopts a seagull.";
    std::vector<llama_token> ptoks(256);
    int np = llama_tokenize(vocab, prompt.c_str(), prompt.size(), ptoks.data(), ptoks.size(), true, false);
    ptoks.resize(np);
    printf("model %s, vocab %d, prompt %d tokens, %d new tokens per sequence\n", model_path.c_str(), n_vocab, np, n_gen);
    printf("%-9s %5s %10s %12s\n", "mode", "batch", "tok/s", "total ms");

    for (int B : batches) {
        double base_tps = 0;
        for (const char * mode : {"default", "logprobs", "coop", "coop-fast", "coop-mt"}) {
            auto cp = llama_context_default_params();
            cp.n_ctx = B * (np + n_gen + 8);
            cp.n_batch = std::max(B * np, 512);
            cp.n_seq_max = B;
            llama_context * ctx = llama_init_from_model(model, cp);
            std::vector<llama_sampler *> smp(B);
            for (int s = 0; s < B; s++) {
                smp[s] = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(smp[s], llama_sampler_init_top_k(40));
                llama_sampler_chain_add(smp[s], llama_sampler_init_top_p(0.95f, 1));
                llama_sampler_chain_add(smp[s], llama_sampler_init_min_p(0.05f, 1));
                llama_sampler_chain_add(smp[s], llama_sampler_init_temp(0.8f));
                llama_sampler_chain_add(smp[s], llama_sampler_init_dist(1234 + s));
            }
            // prompt for every sequence in one batch
            llama_batch batch = llama_batch_init(std::max(B * np, B), 0, 1);
            batch.n_tokens = 0;
            for (int s = 0; s < B; s++)
                for (int i = 0; i < np; i++) {
                    int j = batch.n_tokens++;
                    batch.token[j] = ptoks[i]; batch.pos[j] = i; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = s;
                    batch.logits[j] = (i == np - 1);
                }
            llama_decode(ctx, batch);
            std::vector<int> row(B);
            for (int s = 0; s < B; s++) row[s] = s * np + np - 1;
            std::vector<std::pair<int, double>> top;
            std::vector<int> idx(n_vocab);
            double samp_ms = 0;
            auto t0 = clk::now();
            for (int step = 0; step < n_gen; step++) {
                std::vector<llama_token> next(B);
                auto ts = clk::now();
                if (strcmp(mode, "coop-mt") == 0) {
                    // per-request work spread over CPU threads (requests are independent)
                    std::vector<const float *> lgs(B);
                    for (int s = 0; s < B; s++) lgs[s] = llama_get_logits_ith(ctx, row[s]);
                    const int nt = std::min(B, (int) std::thread::hardware_concurrency());
                    std::vector<std::thread> pool;
                    for (int t = 0; t < nt; t++)
                        pool.emplace_back([&, t]() {
                            std::vector<std::pair<int, double>> tp;
                            for (int s = t; s < B; s += nt) next[s] = coop_choose(lgs[s], n_vocab, s, step, tp);
                        });
                    for (auto & th : pool) th.join();
                    samp_ms += std::chrono::duration<double, std::milli>(clk::now() - ts).count();
                    goto decode;
                }
                for (int s = 0; s < B; s++) {
                    const float * lg = llama_get_logits_ith(ctx, row[s]);
                    if (strcmp(mode, "coop-fast") == 0) {
                        next[s] = coop_choose(lg, n_vocab, s, step, top);
                    } else if (strcmp(mode, "coop") == 0) {
                        for (int v = 0; v < n_vocab; v++) idx[v] = v;
                        top20(lg, n_vocab, top, idx);
                        const double T = 0.8;
                        const std::string key = "bench-request-" + std::to_string(s);
                        int best = -1; double bv = -INFINITY;
                        for (auto & c : top) {
                            double v = c.second / T + gumbel(42, key, step, c.first);
                            if (v > bv || (v == bv && c.first < best)) { bv = v; best = c.first; }
                        }
                        next[s] = best;
                    } else {
                        next[s] = llama_sampler_sample(smp[s], ctx, row[s]);
                        if (strcmp(mode, "logprobs") == 0) {
                            for (int v = 0; v < n_vocab; v++) idx[v] = v;
                            top20(lg, n_vocab, top, idx);
                        }
                    }
                }
                samp_ms += std::chrono::duration<double, std::milli>(clk::now() - ts).count();
            decode:
                batch.n_tokens = 0;
                for (int s = 0; s < B; s++) {
                    int j = batch.n_tokens++;
                    batch.token[j] = next[s]; batch.pos[j] = np + step; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = s;
                    batch.logits[j] = 1;
                    row[s] = j;
                }
                llama_decode(ctx, batch);
            }
            double total = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            double tps = B * n_gen / (total / 1000.0);
            if (strcmp(mode, "default") == 0) base_tps = tps;
            // (sampling time is not reported: decode is asynchronous, so a host-side timer also counts GPU waits)
            printf("%-9s %5d %10.1f %12.0f   (%+.1f%% vs default)\n", mode, B, tps, total, 100 * (tps / base_tps - 1));
            fflush(stdout);
            llama_batch_free(batch);
            for (auto * x : smp) llama_sampler_free(x);
            llama_free(ctx);
        }
    }
    llama_model_free(model);
    return 0;
}
