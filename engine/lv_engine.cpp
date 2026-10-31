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
//
// All logprobs are raw (temperature 1) log-softmax over the full vocabulary, computed in double precision from
// the engine's float logits.

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
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
        float mx = -INFINITY;
        for (int v = 0; v < n_vocab; v++) mx = std::max(mx, lg[v]);
        double sum = 0.0;
        for (int v = 0; v < n_vocab; v++) sum += std::exp((double) lg[v] - mx);
        const double lse = mx + std::log(sum);
        lse_out = lse;
        const int kk = std::min(k, n_vocab);
        // "better" = higher logit, ties to the lower id; the heap keeps the worst of the current best kk on top
        auto better = [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); };
        std::vector<int> heap;
        heap.reserve(kk);
        for (int v = 0; v < n_vocab; v++) {
            if ((int) heap.size() < kk) {
                heap.push_back(v);
                std::push_heap(heap.begin(), heap.end(), better);
            } else if (better(v, heap.front())) {
                std::pop_heap(heap.begin(), heap.end(), better);
                heap.back() = v;
                std::push_heap(heap.begin(), heap.end(), better);
            }
        }
        std::sort(heap.begin(), heap.end(), better);
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

    json start(const std::vector<llama_token> & context, int k) {
        if (context.empty()) throw std::runtime_error("empty context");
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
    std::string model_path, cache_k = "f16", cache_v = "f16";
    int n_ctx = 16384, n_batch = 2048, n_gpu_layers = 99;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-c" && i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        else if (a == "-b" && i + 1 < argc) n_batch = std::stoi(argv[++i]);
        else if (a == "-ngl" && i + 1 < argc) n_gpu_layers = std::stoi(argv[++i]);
        else if (a == "-ctk" && i + 1 < argc) cache_k = argv[++i];  // KV cache types, to emulate a provider that
        else if (a == "-ctv" && i + 1 < argc) cache_v = argv[++i];  // compresses its cache (f16, q8_0, q4_0)
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: lv-engine -m model.gguf [-c ctx] [-b batch] [-ngl layers] [-ctk t] [-ctv t] < requests.jsonl\n");
        return 1;
    }
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
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
    cp.type_k = kv_type(cache_k);
    cp.type_v = kv_type(cache_v);
    if (cache_v != "f16") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // a quantized V cache needs flash attention
    E.ctx = llama_init_from_model(E.model, cp);
    if (!E.ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
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
