// The llmverify public sampling rule inside llama-server (docs/llama-server.md).
//
// A request with "llmverify": {"key": "<client nonce>", "seed": n, "temperature": t, "top_k", "top_p", "min_p"}
// is generated with the rule of llmverify/sampling.py instead of the sampler chain: at every step the top-20
// logprobs the server reports (n_probs = 20, pre-sampling, so they come from exactly the same arithmetic) are
// rounded to 1e-4 and put in canonical order, and the rule picks from them. A proxy (llmverify proxy) turns the
// reply into a signed receipt. The rule code is copied from engine/lv_engine.cpp, which matches the Python bit for bit.
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llmverify {

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

static const int COMMIT_K = 20;

struct verify_params {
    bool enabled = false;
    std::string key;
    Spec spec;
};

// {"key", "seed", "temperature", "top_k", "top_p", "min_p"}; temperature 0 (or absent) means greedy.
static verify_params parse_verify(const nlohmann::json & j) {
    verify_params v;
    if (!j.is_object()) throw std::invalid_argument("must be an object");
    v.key = j.at("key").get<std::string>();
    if (v.key.empty()) throw std::invalid_argument("key must not be empty");
    v.spec.seed = j.value("seed", 0LL);
    v.spec.temperature = j.value("temperature", 0.0);
    v.spec.greedy = !(v.spec.temperature > 0.0);
    v.spec.top_k = j.value("top_k", COMMIT_K);
    v.spec.top_p = j.value("top_p", 1.0);
    v.spec.min_p = j.value("min_p", 0.0);
    if (!std::isfinite(v.spec.temperature) || v.spec.temperature < 0) throw std::invalid_argument("bad temperature");
    if (v.spec.top_k < 1 || v.spec.top_k > COMMIT_K) throw std::invalid_argument("top_k must be in [1, 20]");
    if (!(v.spec.top_p > 0.0 && v.spec.top_p <= 1.0)) throw std::invalid_argument("top_p must be in (0, 1]");
    if (!(v.spec.min_p >= 0.0 && v.spec.min_p < 1.0)) throw std::invalid_argument("min_p must be in [0, 1)");
    v.enabled = true;
    return v;
}

}  // namespace llmverify
