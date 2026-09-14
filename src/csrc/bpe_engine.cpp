#define BUILDING_BPE_DLL
#include "bpe_engine.hpp"

#include <cstring>
#include <cstdlib>
#include <cctype>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

namespace bpe {

// Helper: unescape JSON string
static std::string unescape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next_c = s[i + 1];
            if (next_c == '"') { out += '"'; i++; }
            else if (next_c == '\\') { out += '\\'; i++; }
            else if (next_c == '/') { out += '/'; i++; }
            else if (next_c == 'b') { out += '\b'; i++; }
            else if (next_c == 'f') { out += '\f'; i++; }
            else if (next_c == 'n') { out += '\n'; i++; }
            else if (next_c == 'r') { out += '\r'; i++; }
            else if (next_c == 't') { out += '\t'; i++; }
            else if (next_c == 'u' && i + 5 < s.size()) {
                int cp = 0;
                for (int k = 0; k < 4; ++k) {
                    char h = s[i + 2 + k];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                }
                // Handle surrogate pairs
                if (cp >= 0xD800 && cp <= 0xDBFF && i + 11 < s.size() && s[i + 6] == '\\' && s[i + 7] == 'u') {
                    int cp2 = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i + 8 + k];
                        cp2 <<= 4;
                        if (h >= '0' && h <= '9') cp2 |= (h - '0');
                        else if (h >= 'a' && h <= 'f') cp2 |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp2 |= (h - 'A' + 10);
                    }
                    if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                        cp = 0x10000 + (((cp & 0x3FF) << 10) | (cp2 & 0x3FF));
                        i += 6;
                    }
                }
                if (cp < 0x80) {
                    out += static_cast<char>(cp);
                } else if (cp < 0x800) {
                    out += static_cast<char>(0xC0 | (cp >> 6));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                } else if (cp < 0x10000) {
                    out += static_cast<char>(0xE0 | (cp >> 12));
                    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                } else {
                    out += static_cast<char>(0xF0 | (cp >> 18));
                    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                }
                i += 5;
            } else {
                out += s[i];
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

bool BPEEngine::load_vocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[BPEEngine] Failed to open vocab file: " << vocab_path << std::endl;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    vocab_.clear();
    id_to_vocab_.clear();

    size_t idx = 0;
    size_t len = content.size();
    int max_id = -1;

    // Lightweight streaming JSON parser for string-to-integer dictionary
    while (idx < len) {
        // Find opening quote for key
        while (idx < len && content[idx] != '"') idx++;
        if (idx >= len) break;
        idx++; // skip '"'

        // Read key with escape support
        std::string raw_key;
        while (idx < len) {
            if (content[idx] == '\\' && idx + 1 < len) {
                raw_key += content[idx++];
                raw_key += content[idx++];
            } else if (content[idx] == '"') {
                idx++; // skip closing '"'
                break;
            } else {
                raw_key += content[idx++];
            }
        }

        // Skip colon
        while (idx < len && content[idx] != ':') idx++;
        if (idx >= len) break;
        idx++; // skip ':'

        // Skip whitespace
        while (idx < len && (content[idx] == ' ' || content[idx] == '\t' ||
                             content[idx] == '\r' || content[idx] == '\n')) idx++;
        if (idx >= len) break;

        // Read integer value
        int token_id = 0;
        bool negative = false;
        if (content[idx] == '-') {
            negative = true;
            idx++;
        }
        while (idx < len && content[idx] >= '0' && content[idx] <= '9') {
            token_id = token_id * 10 + (content[idx] - '0');
            idx++;
        }
        if (negative) token_id = -token_id;

        std::string key = unescape_json_string(raw_key);
        vocab_[key] = token_id;
        if (token_id > max_id) {
            max_id = token_id;
        }

        // Register default special tokens if present
        if (key.size() > 2 && key.front() == '<' && key.back() == '>') {
            special_tokens_[key] = token_id;
        }
    }

    if (max_id >= 0) {
        id_to_vocab_.resize(max_id + 1);
        for (const auto& kv : vocab_) {
            if (kv.second >= 0 && kv.second <= max_id) {
                id_to_vocab_[kv.second] = kv.first;
            }
        }
    }

    lru_cache_.clear();
    return !vocab_.empty();
}

bool BPEEngine::load_merges(const std::string& merges_path) {
    std::ifstream file(merges_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[BPEEngine] Failed to open merges file: " << merges_path << std::endl;
        return false;
    }

    bpe_ranks_.clear();
    std::string line;
    int rank = 0;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // skip comments e.g. #version: 0.2
        }

        // Trim carriage return
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos && space_pos > 0 && space_pos + 1 < line.size()) {
            std::string first = line.substr(0, space_pos);
            std::string second = line.substr(space_pos + 1);
            bpe_ranks_[{first, second}] = rank++;
        }
    }

    file.close();
    lru_cache_.clear();
    return !bpe_ranks_.empty();
}

std::vector<int> BPEEngine::encode_word(const std::string& word, float p_dropout) {
    if (word.empty()) {
        return {};
    }

    // Check LRU cache only when deterministic (p_dropout == 0.0)
    if (p_dropout <= 0.0f) {
        std::vector<int> cached;
        if (lru_cache_.get(word, cached)) {
            return cached;
        }
    }

    // Convert raw UTF-8 bytes to mapped Unicode symbols
    std::vector<std::string> symbols;
    symbols.reserve(word.size());
    for (unsigned char b : word) {
        symbols.push_back(byte_map_.byte_to_unicode(b));
    }

    std::vector<int> result;
    int n_symbols = static_cast<int>(symbols.size());

    if (n_symbols == 0) {
        return result;
    }

    if (n_symbols == 1) {
        auto it = vocab_.find(symbols[0]);
        if (it != vocab_.end()) {
            result.push_back(it->second);
        }
        if (p_dropout <= 0.0f) {
            lru_cache_.put(word, result);
        }
        return result;
    }

    // Initialize doubly-linked list node pool
    std::vector<Node> nodes(n_symbols);
    for (int i = 0; i < n_symbols; ++i) {
        nodes[i].str = std::move(symbols[i]);
        nodes[i].prev = i - 1;
        nodes[i].next = i + 1;
    }
    nodes.back().next = -1;

    int head = 0;
    int active_count = n_symbols;

    // Random number generator for BPE-Dropout
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Iteratively merge adjacent pairs
    while (active_count > 1) {
        if (p_dropout > 0.0f) {
            // Stochastic BPE-Dropout
            std::vector<std::pair<int, int>> candidates; // (rank, node_idx)
            int curr = head;
            while (nodes[curr].next != -1) {
                int nxt = nodes[curr].next;
                auto it = bpe_ranks_.find({nodes[curr].str, nodes[nxt].str});
                if (it != bpe_ranks_.end()) {
                    candidates.push_back({it->second, curr});
                }
                curr = nxt;
            }

            if (candidates.empty()) {
                break;
            }

            // Sort candidates by rank ascending
            std::sort(candidates.begin(), candidates.end(),
                      [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                          return a.first < b.first;
                      });

            // Randomly drop candidates with probability p_dropout
            int best_idx = -1;
            for (const auto& cand : candidates) {
                if (dist(rng) >= p_dropout) {
                    best_idx = cand.second;
                    break;
                }
            }

            // Fallback to top candidate if all were dropped
            if (best_idx == -1) {
                best_idx = candidates[0].second;
            }

            // Merge best_idx and next
            int nxt_idx = nodes[best_idx].next;
            nodes[best_idx].str += nodes[nxt_idx].str;
            nodes[best_idx].next = nodes[nxt_idx].next;
            if (nodes[nxt_idx].next != -1) {
                nodes[nodes[nxt_idx].next].prev = best_idx;
            }
            active_count--;
        } else {
            // Deterministic Fast Doubly-Linked List Merge
            int min_rank = INT_MAX;
            int best_idx = -1;
            int curr = head;

            while (nodes[curr].next != -1) {
                int nxt = nodes[curr].next;
                auto it = bpe_ranks_.find({nodes[curr].str, nodes[nxt].str});
                if (it != bpe_ranks_.end() && it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = curr;
                }
                curr = nxt;
            }

            if (best_idx == -1) {
                break; // No more merges available
            }

            int nxt_idx = nodes[best_idx].next;
            nodes[best_idx].str += nodes[nxt_idx].str;
            nodes[best_idx].next = nodes[nxt_idx].next;
            if (nodes[nxt_idx].next != -1) {
                nodes[nodes[nxt_idx].next].prev = best_idx;
            }
            active_count--;
        }
    }

    // Collect final token IDs from active nodes
    result.reserve(active_count);
    int curr = head;
    while (curr != -1) {
        auto it = vocab_.find(nodes[curr].str);
        if (it != vocab_.end()) {
            result.push_back(it->second);
        } else {
            // Fallback: byte-level decomposition ensures 0% UNK guarantee
            for (unsigned char b : nodes[curr].str) {
                auto b_it = vocab_.find(byte_map_.byte_to_unicode(b));
                if (b_it != vocab_.end()) {
                    result.push_back(b_it->second);
                }
            }
        }
        curr = nodes[curr].next;
    }

    if (p_dropout <= 0.0f) {
        lru_cache_.put(word, result);
    }
    return result;
}

// Fast pre-tokenization into word chunks
std::vector<std::string> BPEEngine::split_into_words(const std::string& text) const {
    std::vector<std::string> words;
    if (text.empty()) return words;

    size_t i = 0;
    size_t n = text.size();

    while (i < n) {
        // 1. Check for special tokens first (e.g. <|endoftext|>)
        bool matched_special = false;
        for (const auto& sp : special_tokens_) {
            const std::string& sp_str = sp.first;
            if (i + sp_str.size() <= n && text.compare(i, sp_str.size(), sp_str) == 0) {
                words.push_back(sp_str);
                i += sp_str.size();
                matched_special = true;
                break;
            }
        }
        if (matched_special) continue;

        // 2. Check for English contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if (text[i] == '\'' && i + 1 < n) {
            char c1 = static_cast<char>(std::tolower(text[i + 1]));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                words.push_back(text.substr(i, 2));
                i += 2;
                continue;
            } else if (i + 2 < n) {
                char c2 = static_cast<char>(std::tolower(text[i + 2]));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    words.push_back(text.substr(i, 3));
                    i += 3;
                    continue;
                }
            }
        }

        // 3. Newline sequences (\r\n, \n, \r)
        if (text[i] == '\r' || text[i] == '\n') {
            size_t start = i;
            while (i < n && (text[i] == '\r' || text[i] == '\n')) i++;
            words.push_back(text.substr(start, i - start));
            continue;
        }

        // 4. Consecutive spaces
        if (text[i] == ' ' || text[i] == '\t') {
            size_t start = i;
            while (i < n && (text[i] == ' ' || text[i] == '\t')) i++;
            // If followed by non-whitespace, keep the leading space with the word
            if (i < n && text[i] != '\r' && text[i] != '\n' && (i - start == 1)) {
                // Single space attached to subsequent word
                size_t word_start = start;
                // Accumulate alphanumeric or non-alphanumeric chunk
                if (std::isalnum(static_cast<unsigned char>(text[i])) || (static_cast<unsigned char>(text[i]) >= 0x80)) {
                    while (i < n && (std::isalnum(static_cast<unsigned char>(text[i])) ||
                                     static_cast<unsigned char>(text[i]) >= 0x80)) i++;
                } else {
                    while (i < n && !std::isalnum(static_cast<unsigned char>(text[i])) &&
                           static_cast<unsigned char>(text[i]) < 0x80 &&
                           text[i] != ' ' && text[i] != '\t' && text[i] != '\r' && text[i] != '\n') i++;
                }
                words.push_back(text.substr(word_start, i - word_start));
                continue;
            } else {
                words.push_back(text.substr(start, i - start));
                continue;
            }
        }

        // 5. Alphanumeric sequence or multi-byte UTF-8 sequence
        if (std::isalnum(static_cast<unsigned char>(text[i])) || static_cast<unsigned char>(text[i]) >= 0x80) {
            size_t start = i;
            while (i < n && (std::isalnum(static_cast<unsigned char>(text[i])) ||
                             static_cast<unsigned char>(text[i]) >= 0x80)) {
                i++;
            }
            words.push_back(text.substr(start, i - start));
            continue;
        }

        // 6. Punctuation / symbol sequence
        size_t start = i;
        while (i < n && !std::isalnum(static_cast<unsigned char>(text[i])) &&
               static_cast<unsigned char>(text[i]) < 0x80 &&
               text[i] != ' ' && text[i] != '\t' && text[i] != '\r' && text[i] != '\n') {
            i++;
        }
        words.push_back(text.substr(start, i - start));
    }

    return words;
}

std::vector<int> BPEEngine::encode(const std::string& text, float p_dropout) {
    std::vector<std::string> words = split_into_words(text);
    std::vector<int> all_tokens;
    all_tokens.reserve(words.size() * 2);

    for (const auto& w : words) {
        // Check if word is a special token
        auto sp_it = special_tokens_.find(w);
        if (sp_it != special_tokens_.end()) {
            all_tokens.push_back(sp_it->second);
            continue;
        }

        std::vector<int> word_tokens = encode_word(w, p_dropout);
        all_tokens.insert(all_tokens.end(), word_tokens.begin(), word_tokens.end());
    }

    return all_tokens;
}

std::string BPEEngine::decode(const std::vector<int>& tokens) const {
    std::string out;
    out.reserve(tokens.size() * 4);

    for (int t : tokens) {
        if (t >= 0 && static_cast<size_t>(t) < id_to_vocab_.size()) {
            const std::string& token_str = id_to_vocab_[t];
            // Decode mapped Unicode symbols back into original raw bytes
            out += byte_map_.decode_bytes(token_str);
        }
    }

    return out;
}

#ifdef _WIN32
struct BatchWorkerData {
    BPEEngine* engine;
    const std::vector<std::string>* texts;
    std::vector<std::vector<int>>* results;
    size_t start_idx;
    size_t end_idx;
    float p_dropout;
};

static DWORD WINAPI BatchWorkerThread(LPVOID lpParam) {
    BatchWorkerData* data = reinterpret_cast<BatchWorkerData*>(lpParam);
    for (size_t i = data->start_idx; i < data->end_idx; ++i) {
        (*data->results)[i] = data->engine->encode((*data->texts)[i], data->p_dropout);
    }
    return 0;
}
#endif

std::vector<std::vector<int>> BPEEngine::encode_batch(const std::vector<std::string>& texts,
                                                      float p_dropout,
                                                      int num_threads) {
    std::vector<std::vector<int>> results(texts.size());
    if (texts.empty()) return results;

#ifdef _WIN32
    if (num_threads <= 0) {
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        num_threads = sysinfo.dwNumberOfProcessors;
        if (num_threads < 1) num_threads = 1;
        if (num_threads > 16) num_threads = 16;
    }

    if (num_threads > 1 && texts.size() > 1) {
        size_t total = texts.size();
        size_t chunk = (total + num_threads - 1) / num_threads;
        std::vector<HANDLE> handles;
        std::vector<BatchWorkerData> worker_data(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            size_t s = t * chunk;
            size_t e = (std::min)(s + chunk, total);
            if (s >= total) break;

            worker_data[t] = {this, &texts, &results, s, e, p_dropout};
            HANDLE h = CreateThread(NULL, 0, BatchWorkerThread, &worker_data[t], 0, NULL);
            if (h) handles.push_back(h);
        }

        if (!handles.empty()) {
            WaitForMultipleObjects(static_cast<DWORD>(handles.size()), handles.data(), TRUE, INFINITE);
            for (HANDLE h : handles) CloseHandle(h);
            return results;
        }
    }
#endif

    // Single-threaded fallback
    for (size_t i = 0; i < texts.size(); ++i) {
        results[i] = encode(texts[i], p_dropout);
    }
    return results;
}

} // namespace bpe

// ========================================================
// C-ABI Implementations
// ========================================================
extern "C" {

BPE_API void* bpe_create() {
    return new bpe::BPEEngine();
}

BPE_API void bpe_free(void* engine) {
    if (engine) {
        delete static_cast<bpe::BPEEngine*>(engine);
    }
}

BPE_API int bpe_load_vocab(void* engine, const char* vocab_path) {
    if (!engine || !vocab_path) return 0;
    return static_cast<bpe::BPEEngine*>(engine)->load_vocab(vocab_path) ? 1 : 0;
}

BPE_API int bpe_load_merges(void* engine, const char* merges_path) {
    if (!engine || !merges_path) return 0;
    return static_cast<bpe::BPEEngine*>(engine)->load_merges(merges_path) ? 1 : 0;
}

BPE_API int bpe_get_vocab_size(void* engine) {
    if (!engine) return 0;
    return static_cast<int>(static_cast<bpe::BPEEngine*>(engine)->vocab_size());
}

BPE_API int bpe_encode(void* engine, const char* text, int* out_tokens, int max_tokens, float p_dropout) {
    if (!engine || !text || !out_tokens || max_tokens <= 0) return 0;
    auto tokens = static_cast<bpe::BPEEngine*>(engine)->encode(text, p_dropout);
    int count = (std::min)(static_cast<int>(tokens.size()), max_tokens);
    std::memcpy(out_tokens, tokens.data(), count * sizeof(int));
    return count;
}

BPE_API int bpe_decode(void* engine, const int* tokens, int num_tokens, char* out_buf, int max_buf_len) {
    if (!engine || !tokens || num_tokens <= 0 || !out_buf || max_buf_len <= 0) return 0;
    std::vector<int> tok_vec(tokens, tokens + num_tokens);
    std::string text = static_cast<bpe::BPEEngine*>(engine)->decode(tok_vec);
    int len = (std::min)(static_cast<int>(text.size()), max_buf_len - 1);
    std::memcpy(out_buf, text.data(), len);
    out_buf[len] = '\0';
    return len;
}

BPE_API void* bpe_streaming_create(void* engine) {
    if (!engine) return nullptr;
    auto dec = static_cast<bpe::BPEEngine*>(engine)->create_streaming_decoder();
    return dec.release();
}

BPE_API int bpe_streaming_feed(void* decoder, int token_id, char* out_chunk, int max_chunk_len) {
    if (!decoder || !out_chunk || max_chunk_len <= 0) return 0;
    std::string chunk = static_cast<bpe::StreamingDecoder*>(decoder)->feed(token_id);
    int len = (std::min)(static_cast<int>(chunk.size()), max_chunk_len - 1);
    std::memcpy(out_chunk, chunk.data(), len);
    out_chunk[len] = '\0';
    return len;
}

BPE_API int bpe_streaming_flush(void* decoder, char* out_chunk, int max_chunk_len) {
    if (!decoder || !out_chunk || max_chunk_len <= 0) return 0;
    std::string chunk = static_cast<bpe::StreamingDecoder*>(decoder)->flush();
    int len = (std::min)(static_cast<int>(chunk.size()), max_chunk_len - 1);
    std::memcpy(out_chunk, chunk.data(), len);
    out_chunk[len] = '\0';
    return len;
}

BPE_API void bpe_streaming_free(void* decoder) {
    if (decoder) {
        delete static_cast<bpe::StreamingDecoder*>(decoder);
    }
}

} // extern "C"
