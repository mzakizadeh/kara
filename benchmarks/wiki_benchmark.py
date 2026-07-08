"""
Real-world benchmark: embedding cost of document-collection updates.

Compares three strategies for keeping a RAG vector store in sync as a
document is edited over time, using REAL Wikipedia revision history:

  1. Vanilla       - re-embed the whole document on every update.
  2. LangChain      - REAL LangChain `index()` incremental indexing
                      (RecursiveCharacterTextSplitter + RecordManager dedup).
                      Cost = chunks actually sent to the embedding model.
  3. KARA (ours)    - DAG reuse: re-embed only chunks that cannot be reused.

Y axis = embedding operations (chunks sent to the embedding model).
That is the metric that costs money and latency, measured identically for
all three methods (for LangChain, by counting real embed_documents calls).

Data source: MediaWiki revisions API. No synthetic edits.
"""

import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
import tiktoken
from scipy.interpolate import PchipInterpolator

from kara import OpenAITokenChunker, KARAUpdater

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import InMemoryRecordManager, index
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

API = "https://en.wikipedia.org/w/api.php"
UA = "kara-benchmark/0.1 (research; contact: mzakizadeh.me@gmail.com)"


class CountingEmbeddings(Embeddings):
    """Fake embeddings that count how many texts are actually embedded.

    This is the honest cost metric: LangChain's `index()` only calls
    embed_documents on chunks it decides are new/changed, so the counter
    equals the real number of embedding operations for that method.
    """

    def __init__(self) -> None:
        self.count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.count += len(texts)
        # deterministic, content-dependent vector (values don't matter here)
        return [[float(len(t) % 17), 1.0, 2.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0, 2.0]


def fetch_revisions(title: str, n: int, step: int = 1, from_start: bool = False) -> list[str]:
    """Fetch up to `n` revision texts of an article, oldest->newest.

    `step` keeps every step-th revision (to span more history with fewer points).
    `from_start` pulls the article's EARLIEST revisions (its growth phase, where
    revisions are short and every edit is substantial) instead of the most recent.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    revs: list[dict] = []
    rvcontinue = None
    while len(revs) < n * step + 1:
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "titles": title,
            "rvprop": "content|timestamp|ids",
            "rvslots": "main",
            "rvlimit": "50",
            "formatversion": "2",
            # rvdir=newer walks oldest->newest (growth phase first)
            "rvdir": "newer" if from_start else "older",
        }
        if rvcontinue:
            params["rvcontinue"] = rvcontinue
        r = session.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        page = data["query"]["pages"][0]
        for rev in page.get("revisions", []):
            content = rev["slots"]["main"].get("content", "")
            if content:
                revs.append({"ts": rev["timestamp"], "text": content})
        if "continue" in data:
            rvcontinue = data["continue"]["rvcontinue"]
            time.sleep(0.2)  # be polite to the API
        else:
            break

    if not from_start:
        revs = list(reversed(revs))  # older-direction fetch -> reverse to oldest-first
    sampled = revs[::step][: n + 1]
    print(f"  fetched {len(revs)} revisions, using {len(sampled)} sampled points "
          f"({'earliest' if from_start else 'most recent'})")
    return [r["text"] for r in sampled]


def report_lengths(versions: list[str], enc) -> list[int]:
    """Print revision token lengths compactly so a max-tokens can be chosen."""
    lengths = [len(enc.encode(v)) for v in versions]
    lo, hi = min(lengths), max(lengths)
    mean = sum(lengths) / len(lengths)
    print("\nArticle length (cl100k_base tokens per sampled revision):")
    print(f"  first={lengths[0]}  last={lengths[-1]}  "
          f"min={lo}  max={hi}  mean={mean:.0f}")
    if hi <= 4000:
        print("  -> whole doc fits; no truncation needed (use --max-tokens 0)")
    else:
        print(f"  -> to avoid truncation set --max-tokens {hi}; "
              f"note KARA runtime grows with doc length")
    return lengths


def build_langchain(chunk_tokens: int):
    """Fresh LangChain incremental-indexing stack for one benchmark run."""
    emb = CountingEmbeddings()
    rm = InMemoryRecordManager(namespace="kara-bench")
    rm.create_schema()
    vs = InMemoryVectorStore(emb)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_tokens, chunk_overlap=0
    )
    return emb, rm, vs, splitter


def run(title: str, n_updates: int, step: int, chunk_tokens: int, max_tokens: int,
        from_start: bool):
    print(f"Fetching '{title}' revisions...")
    versions = fetch_revisions(title, n_updates, step, from_start=from_start)
    if len(versions) < 3:
        raise SystemExit("Not enough revisions with content; try another article.")

    enc = tiktoken.get_encoding("cl100k_base")

    # --- change #1: report real article length before anything else ---
    report_lengths(versions, enc)

    # Bound each revision to its first `max_tokens` tokens so the DAG rebuild
    # (O(tokens x chunk_size) hashing per update) stays tractable. This is the
    # lead + first sections of the article as it really evolved -- still real edits.
    if max_tokens:
        versions = [enc.decode(enc.encode(v)[:max_tokens]) for v in versions]
        print(f"Truncating each revision to <= {max_tokens} tokens for the run.")

    chunker = OpenAITokenChunker(encoding_name="cl100k_base", chunk_size=chunk_tokens)
    updater = KARAUpdater(chunker=chunker)

    # --- change #2: real LangChain incremental indexing ---
    emb, rm, vs, splitter = build_langchain(chunk_tokens)

    cum_v = cum_l = cum_k = 0
    vanilla, langchain, kara = [], [], []

    # initial ingest (identical first-time cost for all three: embed everything).
    init = updater.create_collection([versions[0]])
    kara_state = init.new_chunked_doc

    # seed LangChain's record manager with revision 0 (not counted as an update)
    docs0 = splitter.create_documents([versions[0]], metadatas=[{"source": title}])
    index(docs0, rm, vs, cleanup="incremental", source_id_key="source")
    emb.count = 0  # reset so cycle costs start clean

    for i, cur in enumerate(versions[1:], 1):
        # 1. vanilla: full re-embed of every chunk in the new revision
        n_chunks = len(splitter.split_text(cur))
        cum_v += n_chunks

        # 2. langchain: REAL index() call, count chunks actually embedded
        before = emb.count
        docs = splitter.create_documents([cur], metadatas=[{"source": title}])
        res_lc = index(docs, rm, vs, cleanup="incremental", source_id_key="source")
        lc_new = emb.count - before
        cum_l += lc_new

        # 3. kara: reuse via DAG, embed only num_added
        t0 = time.time()
        res = updater.update_collection(kara_state, [cur])
        cum_k += res.num_added
        kara_state = res.new_chunked_doc

        vanilla.append(cum_v)
        langchain.append(cum_l)
        kara.append(cum_k)
        print(
            f"  update {i:>3}: vanilla+{n_chunks:>4}  "
            f"langchain+{lc_new:>4} (add {res_lc['num_added']} upd {res_lc['num_updated']})  "
            f"kara+{res.num_added:>3} (reused {res.num_reused})  "
            f"[{time.time() - t0:.1f}s]"
        )

    plot(title, vanilla, langchain, kara)

    # --- numeric output: print series + write CSV for external plotting ---
    print("\n" + "=" * 60)
    print("RAW DATA (cumulative embedding operations)")
    print("=" * 60)
    print(f"{'cycle':>6} {'vanilla':>10} {'langchain':>10} {'kara':>10}")
    for i in range(len(vanilla)):
        print(f"{i + 1:>6} {vanilla[i]:>10} {langchain[i]:>10} {kara[i]:>10}")

    csv_path = "benchmarks/wiki_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "vanilla", "langchain", "kara"])
        for i in range(len(vanilla)):
            w.writerow([i + 1, vanilla[i], langchain[i], kara[i]])
    print(f"\nWrote {csv_path}")

    print("\nFinal cumulative embedding ops:")
    print(f"  vanilla   : {vanilla[-1]:>6}")
    print(f"  langchain : {langchain[-1]:>6}")
    print(f"  kara      : {kara[-1]:>6}")
    print(f"  KARA saves {100 * (1 - kara[-1] / vanilla[-1]):.1f}% vs vanilla, "
          f"{100 * (1 - kara[-1] / max(langchain[-1], 1)):.1f}% vs langchain")


def _smooth(x, y, n=300):
    """Monotone PCHIP interpolation for smooth, overshoot-free cumulative curves."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return x, y
    xs = np.linspace(x.min(), x.max(), n)
    ys = PchipInterpolator(x, y)(xs)
    return xs, ys


def plot(title, vanilla, langchain, kara):
    """Abstracted 'vitrine' chart: communicate the shape, not the numbers.

    No tick numbers, no legend box, no grid -- just three labeled curves and
    directional axes. Exact figures live in stdout + the CSV.
    """
    x = list(range(1, len(vanilla) + 1))
    series = [
        ("Vanilla", vanilla, "#E4572E"),
        ("LangChain", langchain, "#F3A712"),
        ("KARA", kara, "#2E86AB"),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ymax = max(vanilla[-1], langchain[-1], kara[-1])
    for label, y, color in series:
        xs, ys = _smooth(x, y)
        emphasis = label == "KARA"
        ax.plot(xs, ys, color=color, lw=4.0 if emphasis else 2.5,
                zorder=4 if emphasis else 3, solid_capstyle="round",
                alpha=1.0 if emphasis else 0.85)
        # direct end-of-line label instead of a legend box
        ax.text(x[-1] + 0.4, y[-1], label, color=color, va="center", ha="left",
                fontsize=13, fontweight="bold" if emphasis else "normal")

    # abstract axes: arrows for direction, words instead of numbers
    ax.set_xlabel("Document edits over time  →", fontsize=13, labelpad=8)
    ax.set_ylabel("Re-embedding cost  →", fontsize=13, labelpad=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(1, x[-1] + (x[-1] - 1) * 0.18)  # headroom for end labels
    ax.set_ylim(0, ymax * 1.08)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
        ax.spines[side].set_linewidth(1.2)

    fig.tight_layout()
    out = "benchmarks/wiki_benchmark.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--title", default="Large language model")
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--step", type=int, default=1, help="keep every step-th revision")
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="truncate each revision to this many tokens (0 = full article)")
    p.add_argument("--from-start", action="store_true",
                   help="use the article's earliest revisions (growth phase)")
    a = p.parse_args()
    run(a.title, a.updates, a.step, a.chunk_tokens, a.max_tokens, a.from_start)
