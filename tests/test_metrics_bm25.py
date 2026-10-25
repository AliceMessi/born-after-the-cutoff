import math

from bm25 import BM25, tokenize
from evaluate import bootstrap
from metrics import acc_all, hit_any, mrr, ndcg, report


def test_metrics():
    pred = ["a", "b", "c", "d"]
    assert acc_all(pred, {"a", "c"}, 2) == 0.0 and acc_all(pred, {"a", "c"}, 3) == 1.0
    assert hit_any(pred, {"c"}, 2) == 0.0 and hit_any(pred, {"c"}, 3) == 1.0
    assert mrr(pred, {"c"}) == 1 / 3 and mrr(pred, {"z"}) == 0.0
    assert math.isclose(ndcg(["a", "b"], {"a", "b"}, 2), 1.0)
    assert acc_all(pred, set(), 5) == 0.0  # empty gold never counts as solved
    r = report(pred, {"b"})
    assert r["hit@1"] == 0.0 and r["hit@3"] == 1.0 and r["mrr"] == 0.5


def test_tokenize_splits_identifiers():
    assert tokenize("LitellmModel._fetch_response HTTPServer") == ["litellm", "model", "fetch", "response", "http", "server"]


def test_bm25_ranks_relevant_doc_first():
    docs = ["def fetch_response(): litellm call", "def render_page(): html", "class Tracing: exporter"]
    bm = BM25(docs)
    assert bm.rank("litellm fetch response broken")[0] == 0
    assert bm.rank("tracing exporter crash")[0] == 2


def test_bootstrap_ci_contains_mean():
    rows = [("r1", 1.0), ("r1", 0.0), ("r2", 1.0), ("r3", 0.0), ("r3", 1.0)]
    m, lo, hi = bootstrap(rows, n=200)
    assert lo <= m <= hi and 0 <= lo and hi <= 1
