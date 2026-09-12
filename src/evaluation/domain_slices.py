"""Curated multi-domain evaluation text slices for comprehensive tokenizer benchmarking.

Includes:
1. General Prose (natural English articles, encyclopedic text)
2. Technical (documentation, system manuals, API specs)
3. Scientific (mathematical notation, LaTeX formulas, scientific writing)
4. Code (Python, C++, JavaScript, SQL, indentation, camelCase, snake_case)
5. Numbers (integers, large floats, scientific notation, hex, IP addresses, dates)
6. URLs (domains, paths, query parameters, anchors)
7. Edge Cases (Unicode emojis, accented characters, whitespace blocks)
"""

DOMAIN_PROSE = """
The Amazon rainforest, covering much of northwestern Brazil and extending into Colombia, Peru and other South American countries, is the world's largest tropical rainforest, famed for its biodiversity. It is crisscrossed by thousands of rivers, including the powerful Amazon. The river basin encompasses 7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana.
"""

DOMAIN_TECHNICAL = """
System Architecture and Fault-Tolerant RPC Protocols:
The distributed coordination service employs a Raft consensus algorithm across a cluster of five nodes. Client requests are serialized using Protocol Buffers and transmitted over TLS 1.3 encrypted HTTP/2 connections. When a write operation is initiated, the leader node appends the entry to its write-ahead log (WAL) and concurrently broadcasts an AppendEntries RPC to all follower replicas. Once a quorum of replicas (floor(N/2) + 1) acknowledges persistence to non-volatile storage, the leader commits the transaction and dispatches an asynchronous acknowledgment to the client runtime. Heartbeat timeouts are configured at 150ms with randomized election intervals between 300ms and 600ms.
"""

DOMAIN_SCIENTIFIC = r"""
Quantum Electrodynamics and Statistical Thermodynamics:
The Hamiltonian for the interacting system can be expressed as:
\hat{H} = \int d^3x \left[ \bar{\psi}(x) (i \gamma^\mu \partial_\mu - m) \psi(x) - \frac{1}{4} F_{\mu\nu}(x) F^{\mu\nu}(x) - e \bar{\psi}(x) \gamma^\mu \psi(x) A_\mu(x) \right]
In the canonical ensemble, the partition function Z(\beta) over the discrete spectrum of energy eigenvalues E_n is:
Z(\beta) = \sum_{n=0}^{\infty} g_n e^{-\beta E_n}, \quad \text{where } \beta = \frac{1}{k_B T}.
The Helmholtz free energy follows as F = -\frac{1}{\beta} \ln Z(\beta), yielding the entropy S = -\left( \frac{\partial F}{\partial T} \right)_V.
For continuous limits, the path integral formulation requires evaluating:
\mathcal{Z} = \int \mathcal{D}\bar{\psi} \mathcal{D}\psi \mathcal{D}A \exp\left( i \int d^4x \, \mathcal{L}_{\text{QED}} \right).
"""

DOMAIN_CODE = """
import asyncio
from typing import Optional, Generic, TypeVar

T = TypeVar("T")

class ConcurrentBuffer(Generic[T]):
    def __init__(self, capacity: int = 1024) -> None:
        self._capacity: int = capacity
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=capacity)
        self._is_closed: bool = False

    async def push_item(self, item: T, timeout_seconds: float = 5.0) -> bool:
        if self._is_closed:
            raise RuntimeError("Cannot push to a closed ConcurrentBuffer instance!")
        try:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout_seconds)
            return True
        except asyncio.TimeoutError:
            return False

    async def drain_all(self) -> list[T]:
        items: list[T] = []
        while not self._queue.empty():
            items.append(await self._queue.get())
            self._queue.task_done()
        return items

def quick_sort_indices(arr: list[int], low_idx: int, high_idx: int) -> None:
    if low_idx < high_idx:
        pivot: int = arr[high_idx]
        i: int = low_idx - 1
        for j in range(low_idx, high_idx):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high_idx] = arr[high_idx], arr[i + 1]
        p = i + 1
        quick_sort_indices(arr, low_idx, p - 1)
        quick_sort_indices(arr, p + 1, high_idx)
"""

DOMAIN_NUMBERS = """
Numerical Benchmark Data:
Integers: 42, 100, 1024, 65536, 4294967296, 9223372036854775807, -17, -999999.
Floats and Precision: 3.14159265358979323846, 2.718281828459045, 1.41421356237.
Scientific Notation: 1.054571817e-34, 6.02214076e+23, 2.99792458e8, 6.67430e-11.
Hexadecimal: 0x00, 0xFF, 0x1A2B, 0xDEADBEEF, 0x7FFFFFFF, 0xC0FFEE.
IP Addresses & Subnets: 127.0.0.1, 192.168.1.254, 10.0.0.0/24, 255.255.255.0.
Dates & Timestamps: 2026-09-12, 1999-12-31T23:59:59Z, 1726147200.0.
"""

DOMAIN_URLS = """
Web References & Deep Links:
https://huggingface.co/datasets/HuggingFaceFW/fineweb?sample=sample-10BT&view=viewer#train-partition
http://www.subdomain.example.org:8080/api/v2/search?query=byte+level+bpe&limit=50&offset=100&format=json#results
https://github.com/openai/tiktoken/blob/main/tiktoken/core.py?plain=1#L45-L89
ftp://ftp.archive.org/pub/datasets/corpora/wikitext-103-raw-v1.tar.gz
git+ssh://git@gitlab.internal.corp:2222/nlp-infra/bpe-tokenizer.git@v0.1.0
https://en.wikipedia.org/wiki/Byte_pair_encoding?action=edit&section=3
"""

DOMAIN_EDGE_CASES = """
Edge Cases, Unicode & Emojis:
Accents & Diacritics: François, résumé, über, façade, piñata, cañón, niño.
Multilingual Scripts:
Devanagari: कृत्रिम बुद्धिमत्ता और प्राकृतिक भाषा प्रसंस्करण।
Chinese: 深度学习与大规模语言模型的分词算法。
Arabic: معالجة اللغات الطبيعية باستخدام خوارزمية التجزئة.
Emojis: 🚀 🤖 🔥 🐍 🌟 💻 ⚡ 🧠 📦 📊
Whitespace Sequences: "   ", "\t\t\t", "\n\n\n\n", " \t \n \t ".
Punctuation: !=, ==, ===, !==, <=, >=, ->, =>, :=, <>, ::, ... , /* */, <!-- -->.
"""


def get_standard_domain_slices() -> dict[str, str]:
    """Returns a dictionary containing standardized test text for each domain."""
    return {
        "prose": DOMAIN_PROSE.strip(),
        "technical": DOMAIN_TECHNICAL.strip(),
        "scientific": DOMAIN_SCIENTIFIC.strip(),
        "code": DOMAIN_CODE.strip(),
        "numbers": DOMAIN_NUMBERS.strip(),
        "urls": DOMAIN_URLS.strip(),
        "edge_cases": DOMAIN_EDGE_CASES.strip(),
    }
