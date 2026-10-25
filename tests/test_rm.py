import tempfile
from pathlib import Path

from baseline_rm import issue_identifiers, rank_repo_map
from test_graph import make_repo


def test_identifiers():
    toks = issue_identifiers("Calling `Child.run()` in pkg/core.py breaks; helper_fn and HTTPAdapter fail, see util.fmt")
    assert "Child.run" in toks and "pkg/core.py" in toks and "helper_fn" in toks and "HTTPAdapter" in toks
    assert "util.fmt" in toks and "breaks" not in toks


def test_rank_prefers_seeded_neighbourhood():
    with tempfile.TemporaryDirectory() as d:
        g = make_repo(Path(d))
    files, order, n_seeds = rank_repo_map(g, "`fmt` returns the wrong string")
    assert n_seeds >= 1
    assert order[0] == "pkg.util.fmt"
    assert files[0] == "src/pkg/util.py"
    files2, order2, n2 = rank_repo_map(g, "something unrelated happens")
    assert n2 == 0 and order2  # falls back to plain PageRank
