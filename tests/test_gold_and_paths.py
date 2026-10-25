from gold import map_lines, scopes, touched_old_lines
from paths import is_bot, is_core_py, is_test

SRC = '''import os

class A:
    x = 1

    def f(self):
        return 1

    def g(self):
        return 2

def top(a):
    b = a + 1
    return b

@decorator
def deco():
    pass
'''


def test_modified_method_and_function_but_not_new_method():
    patch = '''@@ -6,5 +6,5 @@ class A:
     def f(self):
-        return 1
+        return 11

     def g(self):
         return 2
+
+    def h(self):
+        return 3
@@ -12,3 +15,4 @@ def top(a):
 def top(a):
     b = a + 1
+    b += 1
     return b
'''
    lines, added = touched_old_lines(patch)
    assert sorted(lines) == [7, 13]
    assert map_lines(lines, scopes(SRC)) == (["A.f", "top"], False)
    assert "    def h(self):" in added


def test_class_attribute_change_maps_to_class():
    patch = '''@@ -3,2 +3,2 @@
 class A:
-    x = 1
+    x = 2
'''
    lines, _ = touched_old_lines(patch)
    assert map_lines(lines, scopes(SRC)) == (["A"], False)


def test_module_level_change_is_flagged():
    patch = '''@@ -1,1 +1,2 @@
 import os
+import sys
'''
    lines, _ = touched_old_lines(patch)
    names, module_level = map_lines(lines, scopes(SRC))
    assert names == [] and module_level


def test_decorator_line_belongs_to_function():
    patch = '''@@ -17,2 +17,2 @@
-@decorator
+@decorator(1)
 def deco():
'''
    lines, _ = touched_old_lines(patch)
    assert map_lines(lines, scopes(SRC))[0] == ["deco"]


def test_path_rules():
    assert is_core_py("src/agents/run.py")
    assert not is_core_py("tests/test_run.py")
    assert not is_core_py("src/pkg/tests/helpers.py")
    assert not is_core_py("docs/conf.py")
    assert not is_core_py("examples/demo.py")
    assert not is_core_py("README.md")
    assert is_test("pkg/foo_test.py") and is_test("conftest.py")


def test_bot_detection():
    assert is_bot({"login": "dependabot[bot]", "__typename": "Bot"})
    assert is_bot({"login": "renovate", "__typename": "User"})
    assert is_bot(None)
    assert not is_bot({"login": "alice", "__typename": "User"})


SRC2 = '''def create(x):
    y = x
    return y


class Store:
    def get(self):
        return 1

    def put(self):
        return 2
'''


def _gold(patch, src=SRC2):
    from gold import credit_new_defs, diff_points

    lines, _, new_defs = diff_points(patch)
    sc = scopes(src)
    return sorted(set(map_lines(lines, sc)[0]) | credit_new_defs(new_defs, sc, src.splitlines()))


def test_appended_function_with_multiline_signature_credits_nothing():
    patch = """@@ -2,2 +2,9 @@ def create(x):
     y = x
     return y
+
+
+def parse(
+    data: str,
+) -> dict:
+    return {}
"""
    assert _gold(patch) == []


def test_new_method_credits_class_and_nested_helper_credits_method():
    new_method = """@@ -10,2 +10,5 @@ class Store:
     def put(self):
         return 2
+
+    def delete(self):
+        return 3
"""
    assert _gold(new_method) == ["Store"]
    nested = """@@ -7,2 +7,4 @@ class Store:
     def get(self):
+        def helper():
+            return 0
         return 1
"""
    assert _gold(nested) == ["Store.get"]


def test_multiline_decorator_block_is_not_credited():
    patch = """@@ -10,2 +10,7 @@ class Store:
     def put(self):
         return 2
+
+    @decorator(
+        option=True,
+    )
+    def other(self):
+        return 4
"""
    assert _gold(patch) == ["Store"]


def test_rewrapped_signature_is_a_modification_not_a_new_def():
    patch = """@@ -7,2 +7,4 @@ class Store:
-    def get(self):
+    def get(
+        self,
+    ):
         return 1
"""
    assert _gold(patch) == ["Store.get"]
