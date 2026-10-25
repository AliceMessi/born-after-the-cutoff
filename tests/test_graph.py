import json

import networkx as nx

from build_graph import build, gold_to_ids, module_id
from tools import CodeGraph

FILES = {
    "src/pkg/__init__.py": "",
    "src/pkg/base.py": "class Base:\n    def hello(self):\n        return 1\n",
    "src/pkg/core.py": (
        "from .base import Base\n"
        "from . import util\n"
        "import pkg.util as u\n\n"
        "class Child(Base):\n"
        "    def run(self):\n"
        "        self.helper()\n"
        "        super().hello()\n"
        "        util.fmt(1)\n"
        "        return len([])\n\n"
        "    def helper(self):\n"
        "        return u.fmt(2)\n\n"
        "def make():\n"
        "    return Child()\n"
    ),
    "src/pkg/util.py": "def fmt(x):\n    return str(x)\n",
    "tests/conftest.py": "def fixture():\n    pass\n",
    "other/conftest.py": "def fixture():\n    pass\n",
    "bad.py": "def broken(:\n",
}


def make_repo(tmp_path):
    for rel, src in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    return build(tmp_path, sorted(FILES), repo="o/r", commit="c" * 40)


def edges_of(g, kind):
    return {(e["source"], e["target"]) for e in g["edges"] if e["type"] == kind}


def test_module_ids():
    files = set(FILES) | {"lib/pkg/__init__.py", "lib/pkg/mod.py", "fastmcp_slim/fastmcp/__init__.py",
                          "fastmcp_slim/fastmcp/server/__init__.py", "fastmcp_slim/fastmcp/server/x.py"}
    assert module_id("src/pkg/core.py", files) == "pkg.core"
    assert module_id("src/pkg/__init__.py", files) == "pkg"
    assert module_id("lib/pkg/mod.py", files) == "pkg.mod"
    assert module_id("fastmcp_slim/fastmcp/server/x.py", files) == "fastmcp.server.x"
    assert module_id("strands-py/src/strands/models/llamacpp.py", files) == "strands.models.llamacpp"


def test_schema_and_networkx_roundtrip(tmp_path):
    g = make_repo(tmp_path)
    assert g["directed"] and g["multigraph"]
    assert all({"id", "name", "text"} <= set(n) for n in g["nodes"])
    assert all({"source", "target", "type", "key"} <= set(e) for e in g["edges"])
    h = nx.node_link_graph(json.loads(json.dumps(g)), edges="edges")
    assert h.is_directed() and h.is_multigraph()
    assert g["graph"]["stats"]["unparseable_files"] == 1
    assert g["graph"]["stats"]["module_id_collisions"] == 1  # two conftest.py outside packages


def test_edges(tmp_path):
    g = make_repo(tmp_path)
    assert ("pkg.core", "pkg.core.Child") in edges_of(g, "contains")
    assert ("pkg.core.Child", "pkg.core.Child.run") in edges_of(g, "contains")
    assert ("pkg.core.Child", "pkg.base.Base") in edges_of(g, "inherits")
    calls = edges_of(g, "calls")
    assert ("pkg.core.Child.run", "pkg.core.Child.helper") in calls  # self.method
    assert ("pkg.core.Child.run", "pkg.base.Base.hello") in calls  # super().method
    assert ("pkg.core.Child.run", "pkg.util.fmt") in calls  # from . import util; util.fmt
    assert ("pkg.core.Child.helper", "pkg.util.fmt") in calls  # import pkg.util as u
    assert ("pkg.core.make", "pkg.core.Child") in calls  # constructor
    assert ("pkg.core", "pkg.base.Base") in edges_of(g, "imports")
    assert g["graph"]["stats"]["calls_builtin"] >= 1  # len()


def test_gold_to_ids(tmp_path):
    g = make_repo(tmp_path)
    assert gold_to_ids(["src/pkg/core.py::Child.run", "src/pkg/core.py::Nope"], g) == ["pkg.core.Child.run", None]


def test_tools(tmp_path):
    cg = CodeGraph(make_repo(tmp_path))
    nb = json.loads(cg.get_code_neighbors("pkg.util.fmt"))
    assert {x["id"] for x in nb["incoming"]} >= {"pkg.core.Child.run", "pkg.core.Child.helper"}
    only_calls = json.loads(cg.get_code_neighbors("Child.run", edge_type="calls"))  # suffix resolution
    assert only_calls["outgoing"] and all(x["edge"] == "calls" for x in only_calls["outgoing"])
    sub = json.loads(cg.get_code_subgraph(["pkg.core.Child", "pkg.base.Base", "missing.x"]))
    assert {"source": "pkg.core.Child", "target": "pkg.base.Base", "edge": "inherits"} in sub["edges"]
    assert sub["unknown"] == ["missing.x"]
    res = json.loads(cg.search_similar_code("fmt helper returns str", k=3))
    assert res["mode"] == "bm25-fallback" and res["results"][0]["id"] == "pkg.util.fmt"
    assert "error" in json.loads(cg.get_code_neighbors("does.not.exist"))
