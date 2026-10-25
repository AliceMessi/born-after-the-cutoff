from baselines import _units
from build_graph import extract
from gold import map_lines, scopes, touched_old_lines

SRC = '''try:
    from fast import Proxy
except ImportError:
    class Proxy:
        def connect(self):
            return 1

if True:
    def helper():
        return 2

class Session:
    if True:
        def maybe(self):
            return 3

    def plain(self):
        def nested():
            return 4
        return nested()
'''


def test_scopes_see_conditional_defs_and_methods():
    names = {s[2] for s in scopes(SRC)}
    assert {"Proxy", "Proxy.connect", "helper", "Session", "Session.maybe", "Session.plain"} <= names
    assert "nested" not in names and "Session.plain.nested" not in names  # nested defs roll up


def test_gold_maps_change_inside_conditional_class():
    patch = '''@@ -5,2 +5,2 @@ except ImportError:
         def connect(self):
-            return 1
+            return 10
'''
    lines, _ = touched_old_lines(patch)
    assert map_lines(lines, scopes(SRC))[0] == ["Proxy.connect"]


def test_units_and_graph_agree_with_gold():
    unit_names = {q for q, _ in _units(SRC)}
    graph = extract("m.py", SRC)
    graph_names = {c["name"] for c in graph["classes"]} | {d["name"] for d in graph["defs"]}
    assert unit_names == graph_names == {s[2] for s in scopes(SRC)}
    assert ("import", "fast", None, None, 0) not in graph["imports"]
    assert ("from", "Proxy", None, "fast", 0) in graph["imports"]
