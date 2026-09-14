#ifndef BPE_ENGINE_HPP
#define BPE_ENGINE_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <climits>
#include <utility>
#include <list>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#ifdef _WIN32
  #include <windows.h>
  #ifdef BUILDING_BPE_DLL
    #define BPE_API __declspec(dllexport)
  #else
    #define BPE_API __declspec(dllimport)
  #endif
#else
  #include <mutex>
  #define BPE_API __attribute__((visibility("default")))
#endif

namespace bpe {

// Cross-platform lightweight mutex
class FastMutex {
public:
#ifdef _WIN32
    FastMutex() { InitializeCriticalSection(&cs_); }
    ~FastMutex() { DeleteCriticalSection(&cs_); }
    void lock() { EnterCriticalSection(&cs_); }
    void unlock() { LeaveCriticalSection(&cs_); }
private:
    CRITICAL_SECTION cs_;
#else
    FastMutex() = default;
    void lock() { m_.lock(); }
    void unlock() { m_.unlock(); }
private:
    std::mutex m_;
#endif
};

class FastLockGuard {
public:
    explicit FastLockGuard(FastMutex& m) : m_(m) { m_.lock(); }
    ~FastLockGuard() { m_.unlock(); }
private:
    FastMutex& m_;
    FastLockGuard(const FastLockGuard&) = delete;
    FastLockGuard& operator=(const FastLockGuard&) = delete;
};

// ========================================================
// 1. Fast 64-bit Pair Hash (Fibonacci Golden Ratio)
// ========================================================
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // Bit-shift with 64-bit golden ratio constant (Knuth's multiplicative hash)
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

// ========================================================
// 2. Doubly-Linked List Node Pool
// ========================================================
struct Node {
    std::string str;
    int prev;
    int next;
};

// ========================================================
// 3. Thread-Safe Word-Level LRU Cache
// ========================================================
class LRUCache {
public:
    explicit LRUCache(size_t capacity = 65536) : capacity_(capacity) {}

    bool get(const std::string& key, std::vector<int>& out_val) {
        FastLockGuard lock(mutex_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return false;
        }
        // Move accessed key to front of LRU list
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second.list_it);
        out_val = it->second.tokens;
        return true;
    }

    void put(const std::string& key, const std::vector<int>& val) {
        FastLockGuard lock(mutex_);
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            it->second.tokens = val;
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.list_it);
            return;
        }

        // Evict oldest if capacity exceeded
        if (cache_map_.size() >= capacity_) {
            const std::string& oldest_key = lru_list_.back();
            cache_map_.erase(oldest_key);
            lru_list_.pop_back();
        }

        lru_list_.push_front(key);
        cache_map_[key] = {val, lru_list_.begin()};
    }

    void clear() {
        FastLockGuard lock(mutex_);
        cache_map_.clear();
        lru_list_.clear();
    }

    size_t size() const {
        return cache_map_.size();
    }

private:
    struct CacheItem {
        std::vector<int> tokens;
        std::list<std::string>::iterator list_it;
    };

    size_t capacity_;
    std::list<std::string> lru_list_;
    std::unordered_map<std::string, CacheItem> cache_map_;
    mutable FastMutex mutex_;
};

// ========================================================
// 4. Bijective 256-Byte Unicode Tables
// ========================================================
class ByteMappingTable {
public:
    ByteMappingTable() {
        init();
    }

    const std::string& byte_to_unicode(uint8_t b) const {
        return b2u_[b];
    }

    // Convert raw byte sequence to mapped Unicode string
    std::string encode_bytes(const std::string& raw) const {
        std::string out;
        out.reserve(raw.size() * 2);
        for (uint8_t b : raw) {
            out += b2u_[b];
        }
        return out;
    }

    // Convert mapped Unicode string back to raw bytes
    std::string decode_bytes(const std::string& mapped_str) const {
        std::string raw;
        raw.reserve(mapped_str.size());
        size_t i = 0;
        while (i < mapped_str.size()) {
            uint8_t c = static_cast<uint8_t>(mapped_str[i]);
            if (c < 0x80) {
                // 1-byte ASCII
                std::string s(1, mapped_str[i]);
                auto it = u2b_.find(s);
                if (it != u2b_.end()) {
                    raw.push_back(static_cast<char>(it->second));
                } else {
                    raw.push_back(mapped_str[i]);
                }
                i += 1;
            } else {
                // 2-byte UTF-8 Unicode character (e.g. 0xC4 0x80)
                if (i + 1 < mapped_str.size()) {
                    std::string s = mapped_str.substr(i, 2);
                    auto it = u2b_.find(s);
                    if (it != u2b_.end()) {
                        raw.push_back(static_cast<char>(it->second));
                        i += 2;
                        continue;
                    }
                }
                // Fallback for single byte if not matched
                raw.push_back(mapped_str[i]);
                i += 1;
            }
        }
        return raw;
    }

private:
    std::string b2u_[256];
    std::unordered_map<std::string, uint8_t> u2b_;

    static std::string codepoint_to_utf8(int cp) {
        std::string s;
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return s;
    }

    void init() {
        std::vector<int> bs;
        for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
        for (int i = 161; i <= 172; ++i) bs.push_back(i);
        for (int i = 174; i <= 255; ++i) bs.push_back(i);

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool found = false;
            for (int x : bs) {
                if (x == b) { found = true; break; }
            }
            if (!found) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        for (size_t i = 0; i < bs.size(); ++i) {
            std::string utf8_char = codepoint_to_utf8(cs[i]);
            b2u_[bs[i]] = utf8_char;
            u2b_[utf8_char] = static_cast<uint8_t>(bs[i]);
        }
    }
};

// ========================================================
// 5. Incremental Streaming UTF-8 Decoder
// ========================================================
class StreamingDecoder {
public:
    StreamingDecoder(const std::vector<std::string>& id_to_token,
                     const ByteMappingTable& byte_map)
        : id_to_token_(id_to_token), byte_map_(byte_map) {}

    // Feed a token and return any newly completed, fully formed UTF-8 text chunk
    std::string feed(int token_id) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= id_to_token_.size()) {
            return "";
        }

        const std::string& token_str = id_to_token_[token_id];
        // Convert token's mapped Unicode symbols back into raw UTF-8 bytes
        std::string raw_bytes = byte_map_.decode_bytes(token_str);

        // Accumulate into pending buffer
        for (char c : raw_bytes) {
            pending_bytes_.push_back(static_cast<uint8_t>(c));
        }

        return extract_valid_utf8(false);
    }

    // Flush any trailing bytes held in the buffer (e.g. at end of stream)
    std::string flush() {
        return extract_valid_utf8(true);
    }

    void reset() {
        pending_bytes_.clear();
    }

private:
    const std::vector<std::string>& id_to_token_;
    const ByteMappingTable& byte_map_;
    std::vector<uint8_t> pending_bytes_;

    std::string extract_valid_utf8(bool is_flush) {
        if (pending_bytes_.empty()) {
            return "";
        }

        size_t valid_end = 0;
        size_t i = 0;
        while (i < pending_bytes_.size()) {
            uint8_t b = pending_bytes_[i];
            size_t seq_len = 0;

            if ((b & 0x80) == 0x00) {
                seq_len = 1; // 1-byte ASCII (0xxxxxxx)
            } else if ((b & 0xE0) == 0xC0) {
                seq_len = 2; // 2-byte sequence (110xxxxx)
            } else if ((b & 0xF0) == 0xE0) {
                seq_len = 3; // 3-byte sequence (1110xxxx)
            } else if ((b & 0xF8) == 0xF0) {
                seq_len = 4; // 4-byte sequence (11110xxx)
            } else {
                // Invalid start byte; skip 1 byte
                if (is_flush) {
                    i++;
                    valid_end = i;
                    continue;
                } else {
                    i++;
                    continue;
                }
            }

            if (i + seq_len <= pending_bytes_.size()) {
                // Sequence is complete in buffer; verify continuation bytes
                bool valid = true;
                for (size_t k = 1; k < seq_len; ++k) {
                    if ((pending_bytes_[i + k] & 0xC0) != 0x80) {
                        valid = false;
                        break;
                    }
                }
                if (valid) {
                    i += seq_len;
                    valid_end = i;
                } else {
                    i += 1; // broken byte, advance 1
                    if (is_flush) valid_end = i;
                }
            } else {
                // Incomplete sequence at buffer boundary
                if (is_flush) {
                    // Flush whatever is left
                    valid_end = pending_bytes_.size();
                }
                break;
            }
        }

        if (valid_end == 0) {
            return "";
        }

        std::string emitted(pending_bytes_.begin(), pending_bytes_.begin() + valid_end);
        pending_bytes_.erase(pending_bytes_.begin(), pending_bytes_.begin() + valid_end);
        return emitted;
    }
};

// ========================================================
// 6. Core BPE Engine
// ========================================================
class BPEEngine {
public:
    BPEEngine() : lru_cache_(65536) {}

    // Load vocab.json
    bool load_vocab(const std::string& vocab_path);

    // Load merges.txt
    bool load_merges(const std::string& merges_path);

    // Encode a pre-tokenized word into subword token IDs
    std::vector<int> encode_word(const std::string& word, float p_dropout = 0.0f);

    // High-level text encoding (with regex word splitting)
    std::vector<int> encode(const std::string& text, float p_dropout = 0.0f);

    // Batch encoding
    std::vector<std::vector<int>> encode_batch(const std::vector<std::string>& texts,
                                               float p_dropout = 0.0f,
                                               int num_threads = 0);

    // Decode token IDs back to original text string
    std::string decode(const std::vector<int>& tokens) const;

    // Create an incremental streaming decoder
    std::unique_ptr<StreamingDecoder> create_streaming_decoder() const {
        return std::unique_ptr<StreamingDecoder>(new StreamingDecoder(id_to_vocab_, byte_map_));
    }

    // Accessors
    size_t vocab_size() const { return vocab_.size(); }
    int get_token_id(const std::string& token) const {
        auto it = vocab_.find(token);
        return (it != vocab_.end()) ? it->second : -1;
    }
    const std::string& get_token_str(int id) const {
        static const std::string empty;
        return (id >= 0 && static_cast<size_t>(id) < id_to_vocab_.size()) ? id_to_vocab_[id] : empty;
    }

    void add_special_token(const std::string& token, int id) {
        special_tokens_[token] = id;
    }

    const ByteMappingTable& byte_map() const { return byte_map_; }

private:
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks_;
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_vocab_;
    std::unordered_map<std::string, int> special_tokens_;
    ByteMappingTable byte_map_;
    LRUCache lru_cache_;

    // Fast C++ regex pre-tokenization
    std::vector<std::string> split_into_words(const std::string& text) const;
};

} // namespace bpe

// ========================================================
// 7. C-ABI Interface (for ctypes, DLL, and foreign runtimes)
// ========================================================
extern "C" {
    BPE_API void* bpe_create();
    BPE_API void  bpe_free(void* engine);
    BPE_API int   bpe_load_vocab(void* engine, const char* vocab_path);
    BPE_API int   bpe_load_merges(void* engine, const char* merges_path);
    BPE_API int   bpe_get_vocab_size(void* engine);
    BPE_API int   bpe_encode(void* engine, const char* text, int* out_tokens, int max_tokens, float p_dropout);
    BPE_API int   bpe_decode(void* engine, const int* tokens, int num_tokens, char* out_buf, int max_buf_len);

    // Streaming decoder C-ABI
    BPE_API void* bpe_streaming_create(void* engine);
    BPE_API int   bpe_streaming_feed(void* decoder, int token_id, char* out_chunk, int max_chunk_len);
    BPE_API int   bpe_streaming_flush(void* decoder, char* out_chunk, int max_chunk_len);
    BPE_API void  bpe_streaming_free(void* decoder);
}

#endif // BPE_ENGINE_HPP
