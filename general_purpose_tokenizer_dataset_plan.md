# General-Purpose GPT-Style Tokenizer
## Dataset & Training Corpus Plan

**Target corpus:** 200–250 MB  
**Vocabulary:** 32,000 tokens  
**Tokenizer:** Byte-Level BPE  
**Goal:** General-purpose, GPT-style tokenizer

---

## 1. Recommended Baseline

| Component | Recommendation |
|---|---|
| Training corpus | 200–250 MB of cleaned text |
| Primary corpus | FineWeb sample / selected FineWeb text |
| Optional curated sources | Wikipedia, WikiText-103, OpenWebText, code, technical/scientific text |
| Tokenizer | Byte-Level BPE |
| Vocabulary size | **32,000** |
| Language focus | English-first, general purpose |
| Objective | Tokenizer suitable for GPT-style language-model experiments |

The goal is not to train a language model with this 200–250 MB corpus. The corpus is used to **learn the tokenizer's vocabulary and merge rules**.

---

## 2. Best Dataset Choices

Use one broad web corpus as the backbone and add smaller curated sources when they provide types of text that the main corpus under-represents.

### FineWeb — Primary Choice

**Role:** General web corpus  
**Recommended amount:** ~150–180 MB

FineWeb is the strongest default for this project because it is based on Common Crawl data and includes extensive filtering and deduplication. It provides broad exposure to natural language and web text.

**Use it for:**

- General English prose
- Articles
- Web pages
- Different writing styles
- Numbers and punctuation
- URLs
- Names and rare words
- General vocabulary

**Recommendation:** **Use FineWeb as the primary corpus.**

---

### Wikipedia

**Role:** Clean encyclopedic and factual prose  
**Recommended amount:** ~15–25 MB

Wikipedia provides relatively clean, structured text and is useful for factual and formal language.

**Use it for:**

- Encyclopedic language
- Proper nouns
- Technical terminology
- Formal prose
- Structured writing

**Recommendation:** Use as a small supplement rather than the main corpus.

---

### WikiText-103

**Role:** Controlled, high-quality long-form Wikipedia-derived text  
**Recommended amount:** Optional ~10–20 MB

WikiText-103 is useful for controlled tokenizer experiments and provides clean long-form prose.

A **32k Byte-Level BPE tokenizer trained on WikiText-103** is also a useful reproducible baseline.

However, WikiText-103 is much less diverse than a broad web corpus.

**Recommendation:** Good for a baseline or small supplement, but **do not use it as the only dataset** for your final general-purpose tokenizer.

---

### OpenWebText

**Role:** Broad web text  
**Recommended amount:** Optional ~20–30 MB

OpenWebText provides another source of diverse web text and can be used as an alternative or supplement to FineWeb.

**Use it for:**

- Web-style writing
- Conversational patterns
- General vocabulary
- Diverse online text

**Recommendation:** Useful, but FineWeb should be your first choice for the primary corpus.

---

### Code Dataset

**Role:** Programming languages and symbolic text  
**Recommended amount:** ~15–25 MB

If you want your tokenizer to work reasonably well with code, include some programming text.

**Use it for:**

- Python
- C/C++
- JavaScript/TypeScript
- JSON
- SQL
- Identifiers
- Operators
- Brackets
- Special characters

Code is especially useful because programming text has token patterns that ordinary prose does not.

**Recommendation:** Include ~10% code if your tokenizer is intended to be broadly useful for AI/ML and software-related workloads.

---

### Technical / Scientific Text

**Role:** Specialized terminology and formal writing  
**Recommended amount:** ~10–15 MB

Useful for exposing the tokenizer to:

- Scientific terminology
- Mathematical notation
- Technical words
- Formal writing
- Domain-specific vocabulary

**Recommendation:** Include a small amount rather than making it a major part of the corpus.

---

### TinyStories

**Role:** Educational / controlled language-model experiment  
**Recommended amount:** Not recommended for the main corpus

TinyStories is excellent for demonstrating small language-model training, but it is deliberately narrow.

Its text mainly consists of simple children's stories with constrained vocabulary.

A general-purpose tokenizer needs exposure to:

- Rare words
- Technical terminology
- Numbers
- URLs
- Code
- Complex sentences
- Different writing styles
- Symbols and punctuation

Therefore:

> **Do not use TinyStories as the foundation of a general-purpose tokenizer.**

It can still be useful as a separate evaluation or educational dataset.

---

# 3. Recommended 250 MB Corpus Composition

For your specific goal, this is the recommended first serious configuration:

| Source | Target Size | Approx. Share | Purpose |
|---|---:|---:|---|
| **FineWeb** | **175 MB** | **70%** | Broad vocabulary and natural web language |
| **Wikipedia / WikiText-103** | **25 MB** | **10%** | Clean factual and encyclopedic prose |
| **Code** | **25 MB** | **10%** | Programming syntax and symbols |
| **Technical / scientific** | **12.5 MB** | **5%** | Specialized terminology and notation |
| **OpenWebText / long-form prose** | **12.5 MB** | **5%** | Additional web and long-form variety |
| **Total** | **250 MB** | **100%** | General-purpose tokenizer corpus |

### Simplest Alternative

If managing multiple datasets becomes inconvenient:

> **Use ~250 MB of FineWeb alone for Version 1.**

This is a perfectly reasonable starting point and is preferable to spending too much effort constructing a complicated corpus before you have a working tokenizer.

---

# 4. Why 32k Vocabulary?

Keep **32,000 tokens** for your first tokenizer.

A rough comparison:

| Vocabulary | Assessment |
|---:|---|
| 8k | Probably too small for a general-purpose tokenizer |
| 16k | Reasonable for smaller models |
| **32k** | **Excellent starting point** |
| 50k–65k | Useful for larger/multilingual experiments |
| 100k+ | Usually unnecessary for this project |

32k gives you a good balance between:

- Vocabulary coverage
- Sequence length
- Model embedding/output size
- Subword flexibility
- Ease of experimentation

A larger vocabulary is **not automatically better**.

---

# 5. Recommended Tokenizer Type

## Byte-Level BPE

Use **Byte-Level BPE** for your first implementation.

It is a strong choice for a GPT-style tokenizer because it:

- Can represent arbitrary text
- Handles unseen words through smaller units
- Does not require a traditional unknown-token fallback for arbitrary byte sequences
- Learns frequent subword patterns
- Works well with English and mixed text
- Handles punctuation and symbols naturally

For your project:

> **Byte-Level BPE + 32k vocabulary** is the recommended starting configuration.

---

# 6. Corpus Preparation Principles

When preparing the 200–250 MB tokenizer corpus:

- Use text representative of the text your future model will process.
- Remove obvious duplicate documents.
- Avoid extremely repetitive boilerplate where possible.
- Preserve punctuation.
- Preserve useful whitespace patterns.
- Preserve numbers.
- Preserve URLs.
- Preserve symbols.
- Preserve casing.
- Avoid aggressive normalization.
- Do not train only on one genre.
- Keep a completely separate evaluation corpus.
- Record the exact datasets and sampling proportions.
- Record preprocessing decisions for reproducibility.
- Prefer high-quality and diverse text over simply adding more low-quality web text.

---

# 7. Suggested Experiment Sequence

Instead of immediately building the final tokenizer, run a few controlled experiments.

| Experiment | Corpus | Vocabulary | Purpose |
|---|---|---:|---|
| **Baseline A** | WikiText-103 subset | 32k | Simple controlled baseline |
| **Baseline B** | ~250 MB FineWeb | 32k | Recommended general-purpose baseline |
| **Experiment C** | Mixed 250 MB corpus | 32k | Likely strongest general-purpose version |
| **Experiment D** | Same mixed corpus | 16k / 32k / 50k | Study vocabulary-size trade-offs |

This lets you understand whether the additional datasets actually improve the tokenizer.

---

# 8. What I Would Use

For your first serious tokenizer:

```text
Corpus
├── FineWeb                    ~175 MB
├── Wikipedia / WikiText-103   ~25 MB
├── Code                       ~25 MB
├── Technical/Scientific       ~12.5 MB
└── OpenWebText/Long-form      ~12.5 MB

Total                           ~250 MB

Tokenizer
├── Algorithm: Byte-Level BPE
└── Vocabulary: 32,000
```

### Final Recommendation

**Version 1 — simplest**

> ~250 MB FineWeb → Byte-Level BPE → 32k vocabulary

**Version 2 — recommended final experiment**

> ~175 MB FineWeb + ~25 MB Wikipedia/WikiText + ~25 MB Code + ~12.5 MB Technical/Scientific + ~12.5 MB OpenWebText/long-form → Byte-Level BPE → 32k vocabulary

The second configuration gives your tokenizer much broader exposure while keeping the total training corpus within your 200–250 MB target.

---

# 9. Important: Tokenizer Corpus vs. LLM Training Corpus

The **200–250 MB limit applies only to the corpus used to train the tokenizer**.

It does **not** mean that a GPT model using this tokenizer should be trained on only 200–250 MB.

The typical workflow is:

```text
Tokenizer Training Corpus
        │
        │ ~200–250 MB
        ▼
Train tokenizer
        │
        ▼
32k vocabulary
        │
        ▼
Use tokenizer to encode
much larger model-training corpus
        │
        ▼
Train language model
```

The tokenizer is trained once and then reused to tokenize the much larger corpus used for language-model training.

---

# 10. Dataset References

- **FineWeb:** https://huggingface.co/datasets/HuggingFaceFW/fineweb
- **FineWeb-Edu:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **WikiText:** https://huggingface.co/datasets/Salesforce/wikitext
- **OpenWebText:** https://huggingface.co/datasets/Skylion007/openwebtext
- **TinyStories:** https://huggingface.co/datasets/roneneldan/TinyStories

---

## Final Configuration

| Item | Choice |
|---|---|
| **Corpus size** | **200–250 MB** |
| **Primary dataset** | **FineWeb** |
| **Tokenizer** | **Byte-Level BPE** |
| **Vocabulary** | **32,000** |
| **Language** | English-first |
| **Target** | General-purpose / GPT-style |
| **Recommended first corpus** | ~250 MB FineWeb |
| **Recommended final experiment** | Mixed corpus described above |
