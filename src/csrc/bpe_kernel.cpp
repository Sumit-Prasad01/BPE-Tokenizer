#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <climits>

// ==========================================
// 1. Fast Pair Hash for unordered_map
// ==========================================
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

// ==========================================
// 2. High-Performance C++ BPE Engine
// ==========================================
struct Node {
    std::string str;
    int prev;
    int next;
};

struct BPEEngine {
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks;
    std::unordered_map<std::string, int> vocab;
    std::string byte_to_unicode[256];
    std::unordered_map<std::string, std::vector<int>> raw_word_cache;

    // Encodes a single word using doubly-linked list BPE merging
    const std::vector<int>& encode_word(const std::string& word) {
        auto cache_it = raw_word_cache.find(word);
        if (cache_it != raw_word_cache.end()) {
            return cache_it->second;
        }

        // Convert raw UTF-8 bytes to mapped Unicode symbols
        std::vector<std::string> symbols;
        symbols.reserve(word.size());
        for (unsigned char b : word) {
            symbols.push_back(byte_to_unicode[b]);
        }

        std::vector<int> result;
        int n_symbols = static_cast<int>(symbols.size());

        if (n_symbols == 0) {
            auto it = raw_word_cache.emplace(word, std::move(result));
            return it.first->second;
        }

        if (n_symbols == 1) {
            auto it = vocab.find(symbols[0]);
            if (it != vocab.end()) {
                result.push_back(it->second);
            }
            auto cache_res = raw_word_cache.emplace(word, std::move(result));
            return cache_res.first->second;
        }

        // Initialize doubly-linked list nodes
        std::vector<Node> nodes(n_symbols);
        for (int i = 0; i < n_symbols; ++i) {
            nodes[i].str = std::move(symbols[i]);
            nodes[i].prev = i - 1;
            nodes[i].next = i + 1;
        }
        nodes.back().next = -1;

        int head = 0;
        int active_count = n_symbols;

        // Iteratively merge adjacent pairs with minimum rank
        while (active_count > 1) {
            int min_rank = INT_MAX;
            int best_idx = -1;
            int curr = head;

            while (nodes[curr].next != -1) {
                int nxt = nodes[curr].next;
                auto it = bpe_ranks.find({nodes[curr].str, nodes[nxt].str});
                if (it != bpe_ranks.end() && it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = curr;
                }
                curr = nxt;
            }

            // If no more merges can be applied, stop
            if (best_idx == -1) {
                break;
            }

            // Merge best_idx and its next node
            int nxt_idx = nodes[best_idx].next;
            nodes[best_idx].str += nodes[nxt_idx].str;
            nodes[best_idx].next = nodes[nxt_idx].next;
            if (nodes[nxt_idx].next != -1) {
                nodes[nodes[nxt_idx].next].prev = best_idx;
            }
            active_count--;
        }

        // Collect final token IDs
        result.reserve(active_count);
        int curr = head;
        while (curr != -1) {
            auto it = vocab.find(nodes[curr].str);
            if (it != vocab.end()) {
                result.push_back(it->second);
            }
            curr = nodes[curr].next;
        }

        auto cache_res = raw_word_cache.emplace(word, std::move(result));
        return cache_res.first->second;
    }

    // Encodes a sequence of words
    std::vector<int> encode_words(const std::vector<std::string>& words) {
        std::vector<int> all_tokens;
        all_tokens.reserve(words.size() * 2);
        for (const auto& w : words) {
            const auto& word_tokens = encode_word(w);
            all_tokens.insert(all_tokens.end(), word_tokens.begin(), word_tokens.end());
        }
        return all_tokens;
    }
};

// ==========================================
// 3. Python C-API Extension Object
// ==========================================
typedef struct {
    PyObject_HEAD
    BPEEngine* engine;
} BPEKernelObject;

static void BPEKernel_dealloc(BPEKernelObject* self) {
    delete self->engine;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int BPEKernel_init(BPEKernelObject* self, PyObject* args, PyObject* kwds) {
    PyObject* py_merges = NULL;
    PyObject* py_vocab = NULL;
    PyObject* py_byte_encoder = NULL;

    static char* kwlist[] = {"merges", "vocab", "byte_encoder", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!", kwlist,
                                     &PyList_Type, &py_merges,
                                     &PyDict_Type, &py_vocab,
                                     &PyDict_Type, &py_byte_encoder)) {
        return -1;
    }

    self->engine = new BPEEngine();

    // 1. Load Byte Encoder mappings (0..255 -> Unicode string)
    PyObject* b_key;
    PyObject* b_val;
    Py_ssize_t b_pos = 0;
    while (PyDict_Next(py_byte_encoder, &b_pos, &b_key, &b_val)) {
        long byte_val = PyLong_AsLong(b_key);
        if (byte_val >= 0 && byte_val < 256 && PyUnicode_Check(b_val)) {
            Py_ssize_t size;
            const char* s = PyUnicode_AsUTF8AndSize(b_val, &size);
            if (s) {
                self->engine->byte_to_unicode[byte_val] = std::string(s, size);
            }
        }
    }

    // 2. Load Vocabulary (token_str -> id)
    PyObject* v_key;
    PyObject* v_val;
    Py_ssize_t v_pos = 0;
    while (PyDict_Next(py_vocab, &v_pos, &v_key, &v_val)) {
        if (PyUnicode_Check(v_key) && PyLong_Check(v_val)) {
            Py_ssize_t size;
            const char* s = PyUnicode_AsUTF8AndSize(v_key, &size);
            long id_val = PyLong_AsLong(v_val);
            if (s) {
                self->engine->vocab[std::string(s, size)] = static_cast<int>(id_val);
            }
        }
    }

    // 3. Load Merges [(pair_a, pair_b), ...]
    Py_ssize_t n_merges = PyList_GET_SIZE(py_merges);
    for (Py_ssize_t i = 0; i < n_merges; ++i) {
        PyObject* item = PyList_GET_ITEM(py_merges, i);
        if (PyTuple_Check(item) && PyTuple_GET_SIZE(item) == 2) {
            PyObject* a = PyTuple_GET_ITEM(item, 0);
            PyObject* b = PyTuple_GET_ITEM(item, 1);
            if (PyUnicode_Check(a) && PyUnicode_Check(b)) {
                Py_ssize_t sz_a, sz_b;
                const char* str_a = PyUnicode_AsUTF8AndSize(a, &sz_a);
                const char* str_b = PyUnicode_AsUTF8AndSize(b, &sz_b);
                if (str_a && str_b) {
                    self->engine->bpe_ranks[{std::string(str_a, sz_a), std::string(str_b, sz_b)}] = static_cast<int>(i);
                }
            }
        }
    }

    return 0;
}

static PyObject* BPEKernel_encode_words(BPEKernelObject* self, PyObject* args) {
    PyObject* words_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &words_list)) {
        return NULL;
    }

    Py_ssize_t n_words = PyList_GET_SIZE(words_list);
    std::vector<std::string> words;
    words.reserve(n_words);

    for (Py_ssize_t i = 0; i < n_words; ++i) {
        PyObject* item = PyList_GET_ITEM(words_list, i);
        if (PyUnicode_Check(item)) {
            Py_ssize_t size;
            const char* s = PyUnicode_AsUTF8AndSize(item, &size);
            if (s) {
                words.emplace_back(s, size);
            }
        }
    }

    // Release GIL during C++ BPE merge execution
    std::vector<int> all_tokens;
    Py_BEGIN_ALLOW_THREADS
    all_tokens = self->engine->encode_words(words);
    Py_END_ALLOW_THREADS

    // Construct Python list
    PyObject* result = PyList_New(all_tokens.size());
    if (!result) return NULL;

    for (size_t i = 0; i < all_tokens.size(); ++i) {
        PyObject* v = PyLong_FromLong(all_tokens[i]);
        if (!v) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, v);
    }

    return result;
}

static PyMethodDef BPEKernel_methods[] = {
    {"encode_words", (PyCFunction)BPEKernel_encode_words, METH_VARARGS, "Encode list of words into token IDs"},
    {NULL}
};

static PyTypeObject BPEKernelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "fast_bpe.BPEKernel",             /* tp_name */
    sizeof(BPEKernelObject),          /* tp_basicsize */
    0,                                /* tp_itemsize */
    (destructor)BPEKernel_dealloc,    /* tp_dealloc */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,               /* tp_flags */
    "From-Scratch C++ BPE Kernel",    /* tp_doc */
    0, 0, 0, 0, 0, 0,
    BPEKernel_methods,                /* tp_methods */
    0, 0, 0, 0, 0, 0, 0,
    (initproc)BPEKernel_init,         /* tp_init */
    0,
    PyType_GenericNew,                /* tp_new */
};

static struct PyModuleDef fast_bpe_module = {
    PyModuleDef_HEAD_INIT,
    "fast_bpe",
    "High-Performance Custom C++ BPE Engine",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_fast_bpe(void) {
    PyObject* m;
    if (PyType_Ready(&BPEKernelType) < 0)
        return NULL;

    m = PyModule_Create(&fast_bpe_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&BPEKernelType);
    if (PyModule_AddObject(m, "BPEKernel", (PyObject*)&BPEKernelType) < 0) {
        Py_DECREF(&BPEKernelType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
