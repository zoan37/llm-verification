// lv-engine: a small llama.cpp front end for verification. It loads one model and answers JSON-lines requests
// on stdin, one JSON reply per line on stdout. Python drives it (llmverify/engine.py).
//
//   {"cmd": "meta"}
//     -> {"n_vocab", "n_ctx", "bos", "eos": [ids], "add_bos", "chat_template", "desc", "trunc_grid"}
//   {"cmd": "tokenize", "text": "...", "special": false}
//     -> {"tokens": [ids], "pieces": ["..."]}          (pieces are UTF-8 with invalid bytes replaced)
//   {"cmd": "detokenize", "tokens": [ids]}
//     -> {"text": "..."}
//   {"cmd": "score", "context": [ids], "tokens": [ids], "k": 20, "temperature": 1.0}
//     One forward pass over context + tokens. For every reply position returns
//     {"top": [[id, lp], ...], "lp": raw logprob of the actual token,
//      "lp_t", "entropy", "m2", "mass_above", "tail_mass"}  (at the given temperature, over the full vocab;
//      left out with "stats": false)
//     plus the sampler-truncation statistics: "rank" (tokens strictly likelier than the actual one), "lp_max", and
//     "ex_k" / "ex_p" / "ex_minp": the probability mass a top_k / top_p / min_p sampler would remove here, one
//     entry per setting in TOPK_GRID / TOPP_GRID / MINP_GRID (also listed by "meta").
//   {"cmd": "start", "context": [ids], "k": 20}       -> {"top": [[id, lp], ...]}  (next-token distribution)
//   {"cmd": "step", "token": id, "k": 20}             -> {"top": [...]}            (append one token, predict next)
//     start/step also take optional "u" (uniform in [0,1)) and "temperature": then the reply adds "sampled", an
//     ordinary full-vocabulary sample by inverse CDF, and "sampled_lp", its raw logprob (honest baselines).
//     start/step keep the KV cache between calls: this is how the cooperative provider generates with the
//     public sampling rule applied in Python, one token per round trip.
//   {"cmd": "generate", "requests": [{"context": [ids], "max_tokens": n, "key": "<nonce>", "eos": [ids],
//                                      "spec": {"mode", "temperature", "top_k", "top_p", "min_p", "seed"}}, ...]}
//     -> {"results": [{"tokens": [ids], "tops": [[[id, q], ...20], ...], "finish_reason": "stop"|"length"}]}
//     The cooperative rule run natively (llmverify/sampling.py, docs/receipt-spec.md), many requests in one batch:
//     per step the top-20 raw logprobs, quantized to integers q = round(lp * 1e4) (half to even) and ordered by
//     (q descending, id ascending), then the public rule picks the token. Bit-identical to the Python path for one
//     request; batched requests are honest generations too (batching changes arithmetic, not the rule). Needs -np.
//   {"cmd": "cb_add", "requests": [...as generate...]} -> {"slots": [...]}     continuous batching: prefill now
//   {"cmd": "cb_step", "steps": n} -> {"finished": [{result + "slot"}], "active": m, "free_cells": c}
//     advance every active sequence up to n tokens; new requests may be added between calls.
//
// All logprobs are raw (temperature 1) log-softmax over the full vocabulary, computed in double precision from
// the engine's float logits.

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Sampler settings the truncation statistics are reported for (common provider defaults and their neighbours).
static const int TOPK_GRID[] = {10, 20, 40, 50, 64, 100, 200};
static const double TOPP_GRID[] = {0.8, 0.9, 0.95, 0.98, 0.99};
static const double MINP_GRID[] = {0.01, 0.02, 0.05, 0.1, 0.2};
static const int TOPK_MAX = 200;

using json = nlohmann::json;

// ---- SHA-256 (FIPS 180-4), for the public sampling noise -------------------------------------------------------
namespace sha {
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void digest(const std::string & msg, uint8_t out[32]) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::string m = msg;
    const uint64_t bits = (uint64_t) msg.size() * 8;
    m.push_back((char) 0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; i--) m.push_back((char) (bits >> (8 * i)));
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t) (uint8_t) m[off + 4 * i] << 24 | (uint32_t) (uint8_t) m[off + 4 * i + 1] << 16 |
                   (uint32_t) (uint8_t) m[off + 4 * i + 2] << 8 | (uint32_t) (uint8_t) m[off + 4 * i + 3];
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) out[4 * i + j] = (uint8_t) (h[i] >> (24 - 8 * j));
}
}  // namespace sha

// ---- The public sampling rule (llmverify/sampling.py), bit for bit ---------------------------------------------
struct Spec {
    bool greedy = true;
    double temperature = 0.0, top_p = 1.0, min_p = 0.0;
    int top_k = 20;
    long long seed = 0;
};
struct Cand { int id; long long q; double lp; };  // q = committed integer units of 1e-4 nats, lp = q / 1e4

static double gumbel_noise(long long seed, const std::string & key, int step, int tok) {
    uint8_t h[32];
    sha::digest("llmverify-gumbel-v1:" + std::to_string(seed) + ":" + key + ":" + std::to_string(step) + ":" +
                std::to_string(tok), h);
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | h[i];
    const double u = ((double) v + 0.5) / 18446744073709551616.0;
    return -std::log(-std::log(u));
}

// rounded, canonically ordered top (receipt.round_top): q = round half to even, order (lp desc, id asc)
static std::vector<Cand> round_top(const std::vector<std::pair<int, double>> & raw) {
    std::vector<Cand> out;
    for (auto & r : raw) {
        if (std::isfinite(r.second)) {
            const long long q = (long long) std::nearbyint(r.second * 10000.0);
            out.push_back({r.first, q, (double) q / 10000.0});
        } else {
            out.push_back({r.first, -100000000LL, -1e4});
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const Cand & a, const Cand & b) {
        return a.lp > b.lp || (a.lp == b.lp && a.id < b.id);
    });
    return out;
}

static int rule_choose(const std::vector<Cand> & top, const Spec & sp, const std::string & key, int step) {
    std::vector<Cand> cands(top.begin(), top.begin() + std::min((size_t) sp.top_k, top.size()));
    if (cands.empty()) throw std::runtime_error("no candidates");
    if (sp.greedy) return cands[0].id;
    const double t = sp.temperature, m = cands[0].lp;
    std::vector<double> w;
    double z = 0.0;
    for (auto & c : cands) { w.push_back(std::exp((c.lp - m) / t)); z += w.back(); }
    std::vector<double> probs;
    for (double x : w) probs.push_back(x / z);
    std::vector<std::pair<Cand, double>> kept;
    for (size_t i = 0; i < cands.size(); i++) kept.push_back({cands[i], probs[i]});
    if (sp.min_p > 0) {
        std::vector<std::pair<Cand, double>> k2;
        for (auto & cp : kept) if (cp.second >= sp.min_p * probs[0]) k2.push_back(cp);
        kept = k2;
    }
    if (sp.top_p < 1.0) {
        std::vector<std::pair<Cand, double>> k2;
        double cum = 0.0;
        for (auto & cp : kept) {
            k2.push_back(cp);
            cum += cp.second;
            if (cum >= sp.top_p) break;
        }
        kept = k2;
    }
    int best = -1;
    double bv = -INFINITY;
    for (auto & cp : kept) {
        const double v = cp.first.lp / t + gumbel_noise(sp.seed, key, step, cp.first.id);
        if (v > bv || (v == bv && cp.first.id < best)) { bv = v; best = cp.first.id; }
    }
    return best;
}

struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0, n_ctx = 0, n_batch = 0;
    int n_threads = 8;
    int n_past = 0;  // tokens in the KV cache for start/step
    const float * last_logits = nullptr;

    std::vector<llama_token> tokenize(const std::string & text, bool special) {
        std::vector<llama_token> out(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), false, special);
        if (n < 0) {
            out.resize(-n);
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), false, special);
        }
        out.resize(std::max(n, 0));
        return out;
    }

    std::string piece(llama_token t) {
        char buf[512];
        int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        return n > 0 ? std::string(buf, n) : std::string();
    }

    // Top-k by a single pass with a small min-heap (no shared buffer, so positions can be scored in parallel).
    json topk(const float * lg, int k, double & lse_out) const {
        // One pass keeps the max and the best kk (higher logit first, ties to the lower id: a later, equal value never
        // displaces an earlier one); a second sums exp(x - max) over logits within 30 nats of the max (the rest add
        // < 1e-13 in all) in float, accumulated in double. Logprobs move by ~1e-7, far below the 1e-4 rounding of a
        // commitment; this pass runs at every scored or generated position, so it is kept cheap.
        const int kk = std::min(k, n_vocab);
        std::vector<int> ids(kk);
        std::vector<float> vals(kk);
        int n = 0;
        float mx = -INFINITY;
        for (int v = 0; v < n_vocab; v++) {
            const float x = lg[v];
            if (x > mx) mx = x;
            if (n < kk || x > vals[n - 1]) {
                int p = n < kk ? n++ : kk - 1;
                while (p > 0 && vals[p - 1] < x) { vals[p] = vals[p - 1]; ids[p] = ids[p - 1]; p--; }
                vals[p] = x; ids[p] = v;
            }
        }
        const float cut = mx - 30.0f;
        double sum = 0.0;
        float part = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            if (lg[v] > cut) part += expf(lg[v] - mx);
            if ((v & 1023) == 1023) { sum += part; part = 0.0f; }
        }
        sum += part;
        const double lse = mx + std::log(sum);
        lse_out = lse;
        std::vector<int> heap(ids.begin(), ids.begin() + n);
        json top = json::array();
        for (int t : heap) top.push_back({t, (double) lg[t] - lse});
        return top;
    }

    // Ordinary ancestral sampling over the full vocabulary at temperature `temp`, using the caller's uniform u
    // (inverse CDF in token-id order). Used to produce honest baseline replies, not by the public rule.
    int sample_full(const float * lg, double temp, double u) {
        double mx = -INFINITY;
        for (int v = 0; v < n_vocab; v++) mx = std::max(mx, (double) lg[v] / temp);
        double z = 0.0;
        for (int v = 0; v < n_vocab; v++) z += std::exp((double) lg[v] / temp - mx);
        double acc = 0.0, target = u * z;
        for (int v = 0; v < n_vocab; v++) {
            acc += std::exp((double) lg[v] / temp - mx);
            if (acc >= target) return v;
        }
        return n_vocab - 1;
    }

    json score_step(const float * lg, int k, llama_token actual, double temp, bool stats) const {
        double lse = 0;
        json step;
        step["top"] = topk(lg, k, lse);
        step["lp"] = (double) lg[actual] - lse;
        if (!stats) return step;
        // Stats at temperature `temp`
        double mx = -INFINITY;
        for (int v = 0; v < n_vocab; v++) mx = std::max(mx, (double) lg[v] / temp);
        double z = 0.0;
        for (int v = 0; v < n_vocab; v++) z += std::exp((double) lg[v] / temp - mx);
        const double lse_t = mx + std::log(z);
        const double lp_x = (double) lg[actual] / temp - lse_t;
        double ent = 0.0, m2 = 0.0, above = 0.0, tail = 0.0;
        const double tail_lp = std::log(1e-4), lp_max = mx - lse_t;
        constexpr int NM = sizeof(MINP_GRID) / sizeof(MINP_GRID[0]);
        double ex_minp[NM] = {0};
        double minp_lp[NM];
        for (int m = 0; m < NM; m++) minp_lp[m] = lp_max + std::log(MINP_GRID[m]);
        std::vector<double> probs(n_vocab);
        int rank = 0;
        for (int v = 0; v < n_vocab; v++) {
            const double lp = (double) lg[v] / temp - lse_t;
            const double p = std::exp(lp);
            probs[v] = p;
            ent -= p * lp;
            m2 += p * lp * lp;
            if (lp > lp_x) { above += p; rank++; }
            if (lp < tail_lp) tail += p;
            for (int m = 0; m < NM; m++) if (lp < minp_lp[m]) ex_minp[m] += p;
        }
        step["lp_t"] = lp_x;
        step["entropy"] = ent;
        step["m2"] = m2;
        step["mass_above"] = above;
        step["tail_mass"] = tail;  // total probability of tokens with p < 1e-4 (the long tail)
        step["rank"] = rank;
        step["lp_max"] = lp_max;
        // top_k removes everything past the K most likely tokens; top_p keeps the shortest likeliest prefix whose
        // mass reaches P (the token that crosses P stays), so it removes the tokens whose mass_above is >= P.
        const int kmax = std::min(TOPK_MAX, n_vocab);
        std::partial_sort(probs.begin(), probs.begin() + kmax, probs.end(), std::greater<double>());
        std::vector<double> cum(kmax);
        for (int i = 0; i < kmax; i++) cum[i] = (i ? cum[i - 1] : 0.0) + probs[i];
        json ex_k = json::array(), ex_p = json::array(), ex_mp = json::array();
        for (int K : TOPK_GRID) ex_k.push_back(std::max(0.0, 1.0 - cum[std::min(K, kmax) - 1]));
        bool sorted_all = false;
        for (double P : TOPP_GRID) {
            int j = (int) (std::lower_bound(cum.begin(), cum.end(), P) - cum.begin());
            if (j == kmax && !sorted_all) {  // a flat distribution: the nucleus is longer than TOPK_MAX
                std::sort(probs.begin() + kmax, probs.end(), std::greater<double>());
                cum.resize(n_vocab);
                for (int i = kmax; i < n_vocab; i++) cum[i] = cum[i - 1] + probs[i];
                sorted_all = true;
                j = (int) (std::lower_bound(cum.begin(), cum.end(), P) - cum.begin());
            }
            ex_p.push_back(j >= (int) cum.size() ? 0.0 : std::max(0.0, 1.0 - cum[j]));
        }
        for (int m = 0; m < NM; m++) ex_mp.push_back(ex_minp[m]);
        step["ex_k"] = ex_k;
        step["ex_p"] = ex_p;
        step["ex_minp"] = ex_mp;
        return step;
    }

    json score(const std::vector<llama_token> & context, const std::vector<llama_token> & reply, int k, double temp,
               bool stats) {
        if (context.empty() || reply.empty()) throw std::runtime_error("empty context or reply");
        if (cb_active()) throw std::runtime_error("continuous batch in progress");
        std::vector<llama_token> seq(context);
        seq.insert(seq.end(), reply.begin(), reply.end());
        const int n = (int) seq.size(), first = (int) context.size() - 1;  // logits at i predict seq[i+1]
        if (n > n_ctx) throw std::runtime_error("sequence longer than context");
        llama_memory_clear(llama_get_memory(ctx), true);
        n_past = 0;
        json steps = json::array();
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        for (int start = 0; start < n - 1; start += n_batch) {
            const int end = std::min(start + n_batch, n - 1);
            batch.n_tokens = 0;
            for (int i = start; i < end; i++) {
                const int j = batch.n_tokens++;
                batch.token[j] = seq[i];
                batch.pos[j] = i;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = i >= first;
            }
            if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); throw std::runtime_error("llama_decode failed"); }
            // Score this batch's positions in parallel: each needs passes over the full vocabulary, which dominate
            // the runtime once the forward pass is on a GPU.
            const int lo = std::max(start, first), cnt = end - lo;
            std::vector<json> out(std::max(cnt, 0));
            std::vector<const float *> rows(std::max(cnt, 0));
            for (int i = lo; i < end; i++) rows[i - lo] = llama_get_logits_ith(ctx, i - start);
            const int nt = std::max(1, std::min(n_threads, cnt));
            std::vector<std::thread> pool;
            for (int t = 0; t < nt; t++)
                pool.emplace_back([&, t]() {
                    for (int j = t; j < cnt; j += nt) out[j] = score_step(rows[j], k, seq[lo + j + 1], temp, stats);
                });
            for (auto & th : pool) th.join();
            for (auto & o : out) steps.push_back(std::move(o));
        }
        llama_batch_free(batch);
        return steps;
    }

    // Raw top-k (id, logprob at temperature 1) as doubles, same values topk() reports.
    std::vector<std::pair<int, double>> topk_raw(const float * lg, int k) const {
        double lse;
        json t = topk(lg, k, lse);
        std::vector<std::pair<int, double>> out;
        for (auto & e : t) out.emplace_back(e[0].get<int>(), e[1].get<double>());
        return out;
    }

    template <class F> void parallel_for(int n, F f) const {
        const int nt = std::max(1, std::min(n_threads, n));
        if (nt == 1) { for (int i = 0; i < n; i++) f(i); return; }
        std::vector<std::thread> pool;
        for (int t = 0; t < nt; t++)
            pool.emplace_back([&, t]() { for (int i = t; i < n; i += nt) f(i); });
        for (auto & th : pool) th.join();
    }

    // ---- native cooperative generation ------------------------------------------------------------------------
    struct Seq {
        std::vector<llama_token> ctxt, tokens, eos;
        std::vector<std::vector<Cand>> tops;
        Spec sp; std::string key; int max_tokens = 0, pos = 0, row = -1; bool done = false; std::string finish = "length";
        long long cells() const { return (long long) ctxt.size() + max_tokens; }
    };

    static Seq parse_request(const json & r, int k) {
        Seq q;
        q.ctxt = r.at("context").get<std::vector<llama_token>>();
        q.max_tokens = r.at("max_tokens").get<int>();
        q.key = r.at("key").get<std::string>();
        q.eos = r.value("eos", std::vector<llama_token>{});
        const json & sp = r.at("spec");
        q.sp.greedy = sp.value("mode", std::string("greedy")) == "greedy";
        q.sp.temperature = sp.value("temperature", 0.0);
        q.sp.top_k = sp.value("top_k", 20);
        q.sp.top_p = sp.value("top_p", 1.0);
        q.sp.min_p = sp.value("min_p", 0.0);
        q.sp.seed = sp.value("seed", 0LL);
        if (q.ctxt.empty() || q.max_tokens < 1) throw std::runtime_error("empty context or max_tokens < 1");
        if (q.sp.top_k < 1 || q.sp.top_k > k) throw std::runtime_error("top_k out of range");
        return q;
    }

    static json result(const Seq & q) {
        json tops = json::array();
        for (size_t i = 0; i < q.tokens.size(); i++) {
            json t = json::array();
            for (auto & c : q.tops[i]) t.push_back({c.id, c.q});
            tops.push_back(t);
        }
        return {{"tokens", q.tokens}, {"tops", tops}, {"finish_reason", q.finish}};
    }

    void decode_or_throw(llama_batch & batch) {
        if (batch.n_tokens && llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
    }

    static int batch_add(llama_batch & b, llama_token t, int pos, int seq, bool logits) {
        const int j = b.n_tokens++;
        b.token[j] = t; b.pos[j] = pos; b.n_seq_id[j] = 1; b.seq_id[j][0] = seq; b.logits[j] = logits;
        return j;
    }

    // Feed sequence `slot`'s prompt (in chunks of n_batch) and read its first-step top before anything else decodes.
    void prefill(Seq & q, int slot, int k) {
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        try {
            for (size_t off = 0; off < q.ctxt.size(); off += n_batch) {
                batch.n_tokens = 0;
                const size_t end = std::min(q.ctxt.size(), off + (size_t) n_batch);
                for (size_t i = off; i < end; i++) {
                    const int j = batch_add(batch, q.ctxt[i], (int) i, slot, i + 1 == q.ctxt.size());
                    if (i + 1 == q.ctxt.size()) q.row = j;
                }
                decode_or_throw(batch);
            }
            q.pos = (int) q.ctxt.size();
            q.tops.push_back(round_top(topk_raw(llama_get_logits_ith(ctx, q.row), k)));
        } catch (...) { llama_batch_free(batch); throw; }
        llama_batch_free(batch);
    }

    // One step for the given sequences (seqs[i] lives in KV sequence slots[i]): each picks its token by the rule,
    // the ones that continue are decoded together, and their next tops are read.
    void advance(std::vector<Seq *> & seqs, const std::vector<int> & slots, int k) {
        parallel_for((int) seqs.size(), [&](int i) {
            Seq & q = *seqs[i];
            const int tok = rule_choose(q.tops.back(), q.sp, q.key, (int) q.tokens.size());
            q.tokens.push_back(tok);
            if (std::find(q.eos.begin(), q.eos.end(), tok) != q.eos.end()) { q.done = true; q.finish = "stop"; }
            else if ((int) q.tokens.size() >= q.max_tokens) q.done = true;
        });
        llama_batch batch = llama_batch_init(std::max(n_batch, (int) seqs.size()), 0, 1);
        std::vector<Seq *> fed;
        for (size_t i = 0; i < seqs.size(); i++) {
            Seq & q = *seqs[i];
            if (q.done) continue;
            q.row = batch_add(batch, q.tokens.back(), q.pos++, slots[i], true);
            fed.push_back(&q);
        }
        try { decode_or_throw(batch); } catch (...) { llama_batch_free(batch); throw; }
        llama_batch_free(batch);
        parallel_for((int) fed.size(), [&](int i) {
            fed[i]->tops.push_back(round_top(topk_raw(llama_get_logits_ith(ctx, fed[i]->row), k)));
        });
    }

    json generate(const json & reqs, int k) {
        const int B = (int) reqs.size();
        if (B < 1) throw std::runtime_error("no requests");
        if (B > (int) llama_n_seq_max(ctx)) throw std::runtime_error("more requests than -np sequences");
        if (cb_active()) throw std::runtime_error("continuous batch in progress");
        std::vector<Seq> seqs;
        long long cells = 0;
        for (int s = 0; s < B; s++) { seqs.push_back(parse_request(reqs[s], k)); cells += seqs.back().cells(); }
        if (cells > n_ctx) throw std::runtime_error("requests need more context than -c provides");
        llama_memory_clear(llama_get_memory(ctx), true);
        n_past = 0;
        last_logits = nullptr;  // start/step state is gone
        for (int s = 0; s < B; s++) prefill(seqs[s], s, k);
        for (;;) {
            std::vector<Seq *> live;
            std::vector<int> slots;
            for (int s = 0; s < B; s++) if (!seqs[s].done) { live.push_back(&seqs[s]); slots.push_back(s); }
            if (live.empty()) break;
            advance(live, slots, k);
        }
        json results = json::array();
        for (auto & q : seqs) results.push_back(result(q));
        return results;
    }

    // ---- continuous batching: sequences join and leave between steps ------------------------------------------
    std::vector<Seq> cb;
    std::vector<char> cb_used;
    long long cb_cells = 0;

    bool cb_active() const { return std::find(cb_used.begin(), cb_used.end(), 1) != cb_used.end(); }

    json cb_add(const json & reqs, int k) {
        if (cb_used.empty()) {
            cb.assign(llama_n_seq_max(ctx), Seq());
            cb_used.assign(llama_n_seq_max(ctx), 0);
        }
        if (!cb_active()) {  // first use since start/step or generate: start from an empty cache
            llama_memory_clear(llama_get_memory(ctx), true);
            n_past = 0;
            last_logits = nullptr;
            cb_cells = 0;
        }
        json slots = json::array();
        for (const auto & r : reqs) {
            Seq q = parse_request(r, k);
            int slot = -1;
            for (size_t i = 0; i < cb_used.size(); i++) if (!cb_used[i]) { slot = (int) i; break; }
            if (slot < 0) throw std::runtime_error("no free sequence slot");
            if (cb_cells + q.cells() > n_ctx) throw std::runtime_error("not enough free context");
            prefill(q, slot, k);
            cb[slot] = std::move(q);
            cb_used[slot] = 1;
            cb_cells += cb[slot].cells();
            slots.push_back(slot);
        }
        return slots;
    }

    // Advance every active sequence by up to `steps` tokens; return the ones that finished and free their cells.
    json cb_step(int steps, int k) {
        json finished = json::array();
        for (int n = 0; n < steps; n++) {
            std::vector<Seq *> live;
            std::vector<int> slots;
            for (size_t i = 0; i < cb_used.size(); i++) if (cb_used[i]) { live.push_back(&cb[i]); slots.push_back((int) i); }
            if (live.empty()) break;
            advance(live, slots, k);
            for (size_t i = 0; i < live.size(); i++) {
                if (!live[i]->done) continue;
                json r = result(*live[i]);
                r["slot"] = slots[i];
                finished.push_back(r);
                llama_memory_seq_rm(llama_get_memory(ctx), slots[i], -1, -1);
                cb_cells -= live[i]->cells();
                cb_used[slots[i]] = 0;
                cb[slots[i]] = Seq();
            }
        }
        return {{"finished", finished}, {"active", (int) std::count(cb_used.begin(), cb_used.end(), 1)},
                {"free_cells", n_ctx - cb_cells}};
    }

    json start(const std::vector<llama_token> & context, int k) {
        if (context.empty()) throw std::runtime_error("empty context");
        if (cb_active()) throw std::runtime_error("continuous batch in progress");
        if ((int) context.size() >= n_ctx) throw std::runtime_error("context too long");
        llama_memory_clear(llama_get_memory(ctx), true);
        // decode in batches; only the final token's logits are needed
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        const int n = (int) context.size();
        for (int s = 0; s < n; s += n_batch) {
            const int e = std::min(n, s + n_batch);
            batch.n_tokens = 0;
            for (int i = s; i < e; i++) {
                const int j = batch.n_tokens++;
                batch.token[j] = context[i];
                batch.pos[j] = i;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = (i == n - 1);
            }
            if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); throw std::runtime_error("llama_decode failed"); }
        }
        const int last = batch.n_tokens - 1;
        llama_batch_free(batch);
        n_past = n;
        last_logits = llama_get_logits_ith(ctx, last);
        double lse;
        return topk(last_logits, k, lse);
    }

    json step(llama_token tok, int k) {
        if (n_past == 0) throw std::runtime_error("step before start");
        if (n_past + 1 >= n_ctx) throw std::runtime_error("context full");
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = tok;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); throw std::runtime_error("llama_decode failed"); }
        llama_batch_free(batch);
        n_past++;
        last_logits = llama_get_logits_ith(ctx, 0);
        double lse;
        return topk(last_logits, k, lse);
    }
};

static std::string meta_str(llama_model * model, const char * key) {
    char buf[1 << 16];
    int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return n >= 0 ? std::string(buf, std::min(n, (int) sizeof(buf) - 1)) : std::string();
}

int main(int argc, char ** argv) {
    std::string model_path, cache_k = "f16", cache_v = "f16", flash = "auto";
    int n_ctx = 16384, n_batch = 2048, n_gpu_layers = 99, n_seq = 1;
    bool vocab_only = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-c" && i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        else if (a == "-b" && i + 1 < argc) n_batch = std::stoi(argv[++i]);
        else if (a == "-ngl" && i + 1 < argc) n_gpu_layers = std::stoi(argv[++i]);
        else if (a == "-ctk" && i + 1 < argc) cache_k = argv[++i];  // KV cache types, to emulate a provider that
        else if (a == "-ctv" && i + 1 < argc) cache_v = argv[++i];  // compresses its cache (f16, q8_0, q4_0)
        else if (a == "-np" && i + 1 < argc) n_seq = std::stoi(argv[++i]);  // parallel sequences for "generate"
        else if (a == "-fa" && i + 1 < argc) flash = argv[++i];  // flash attention: on, off or auto (default)
        else if (a == "-vocab") vocab_only = true;  // tokenizer only (meta, tokenize, detokenize): no weights, no context
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: lv-engine -m model.gguf [-c ctx] [-b batch] [-ngl layers] [-ctk t] [-ctv t] [-np seqs] [-vocab] < requests.jsonl\n");
        return 1;
    }
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.vocab_only = vocab_only;
    Engine E;
    E.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!E.model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = std::min(n_batch, 512);
    auto kv_type = [](const std::string & t) {
        if (t == "q8_0") return GGML_TYPE_Q8_0;
        if (t == "q4_0") return GGML_TYPE_Q4_0;
        if (t == "f16") return GGML_TYPE_F16;
        throw std::runtime_error("unknown cache type " + t);
    };
    cp.n_seq_max = n_seq;
    cp.kv_unified = true;  // sequences share the n_ctx cells
    cp.type_k = kv_type(cache_k);
    cp.type_v = kv_type(cache_v);
    if (cache_v != "f16") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // a quantized V cache needs flash attention
    if (flash == "on") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    else if (flash == "off") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    E.ctx = vocab_only ? nullptr : llama_init_from_model(E.model, cp);
    if (!E.ctx && !vocab_only) { fprintf(stderr, "failed to create context\n"); return 1; }
    E.vocab = llama_model_get_vocab(E.model);
    E.n_vocab = llama_vocab_n_tokens(E.vocab);
    E.n_ctx = n_ctx;
    E.n_batch = n_batch;
    E.n_threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    fprintf(stderr, "lv-engine ready: vocab %d, ctx %d\n", E.n_vocab, n_ctx);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json req, resp;
        try {
            req = json::parse(line);
            resp["id"] = req.value("id", json());
            const std::string cmd = req.value("cmd", "score");
            const int k = req.value("k", 20);
            if (cmd == "meta") {
                char desc[256];
                llama_model_desc(E.model, desc, sizeof(desc));
                json eos = json::array();
                for (int t = 0; t < E.n_vocab; t++)
                    if (llama_vocab_is_eog(E.vocab, t)) eos.push_back(t);
                resp["n_vocab"] = E.n_vocab;
                resp["n_ctx"] = E.n_ctx;
                resp["bos"] = llama_vocab_bos(E.vocab);
                resp["add_bos"] = llama_vocab_get_add_bos(E.vocab);
                resp["eos"] = eos;
                resp["chat_template"] = meta_str(E.model, "tokenizer.chat_template");
                resp["desc"] = desc;
                resp["features"] = E.ctx ? json::array({"generate"}) : json::array();
                resp["n_seq_max"] = E.ctx ? (int) llama_n_seq_max(E.ctx) : 0;
                resp["vocab_only"] = E.ctx == nullptr;
                resp["trunc_grid"] = {{"top_k", std::vector<int>(std::begin(TOPK_GRID), std::end(TOPK_GRID))},
                                      {"top_p", std::vector<double>(std::begin(TOPP_GRID), std::end(TOPP_GRID))},
                                      {"min_p", std::vector<double>(std::begin(MINP_GRID), std::end(MINP_GRID))}};
            } else if (cmd == "tokenize") {
                auto ids = E.tokenize(req.value("text", ""), req.value("special", false));
                json pieces = json::array();
                for (auto t : ids) pieces.push_back(E.piece(t));
                resp["tokens"] = ids;
                resp["pieces"] = pieces;
            } else if (cmd == "detokenize") {
                std::string out;
                for (int t : req["tokens"].get<std::vector<int>>()) out += E.piece(t);
                resp["text"] = out;
            } else if (cmd == "score") {
                auto context = req["context"].get<std::vector<llama_token>>();
                auto reply = req["tokens"].get<std::vector<llama_token>>();
                resp["steps"] = E.score(context, reply, k, req.value("temperature", 1.0), req.value("stats", true));
            } else if (!E.ctx && cmd != "tokenize" && cmd != "detokenize") {
                throw std::runtime_error("vocab-only engine: only meta, tokenize and detokenize");
            } else if (cmd == "generate") {
                resp["results"] = E.generate(req.at("requests"), k);
            } else if (cmd == "cb_add") {
                resp["slots"] = E.cb_add(req.at("requests"), k);
            } else if (cmd == "cb_step") {
                resp.update(E.cb_step(req.value("steps", 1), k));
            } else if (cmd == "start" || cmd == "step") {
                resp["top"] = cmd == "start" ? E.start(req["context"].get<std::vector<llama_token>>(), k)
                                             : E.step(req["token"].get<llama_token>(), k);
                if (req.contains("u")) {  // also draw an ordinary full-vocabulary sample
                    const int t = E.sample_full(E.last_logits, req.value("temperature", 1.0), req["u"].get<double>());
                    double lse;
                    E.topk(E.last_logits, 1, lse);
                    resp["sampled"] = t;
                    resp["sampled_lp"] = (double) E.last_logits[t] - lse;
                }
            } else {
                throw std::runtime_error("unknown cmd " + cmd);
            }
        } catch (const std::exception & e) {
            resp["error"] = e.what();
        }
        std::cout << resp.dump(-1, ' ', false, json::error_handler_t::replace) << "\n" << std::flush;
    }
    llama_free(E.ctx);
    llama_model_free(E.model);
    llama_backend_free();
    return 0;
}
