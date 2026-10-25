from run_g2 import parse_action


def test_parse_action_variants():
    assert parse_action('{"tool": "search_similar_code", "args": {"query": "retry", "k": 5}}')["tool"] == "search_similar_code"
    assert parse_action('Sure!\n```json\n{"final": ["a.py::f"]}\n```\nDone.')["final"] == ["a.py::f"]
    assert parse_action('{"tool": "read_file", "args": {"path": "x.py"}} trailing {text}')["args"]["path"] == "x.py"
    assert parse_action("no json here") is None
    assert parse_action('["not", "a", "dict"]') is None


def test_function_calling_aliases():
    a = parse_action('{"name": "get_code_neighbors", "arguments": {"node": "A.b"}}')
    assert a == {"tool": "get_code_neighbors", "args": {"node": "A.b"}}
    b = parse_action('{"function": {"name": "read_file", "arguments": "{\\"path\\": \\"x.py\\"}"}}')
    assert b["tool"] == "read_file" and b["args"] == {"path": "x.py"}
    assert parse_action('{"final": []}') == {"final": []}


def test_openai_style_tool_calls_list_from_pilot():
    raw = ('```json\n{\n  "tool_calls": [\n    {\n      "tool_name": "read_file",\n      "args": {\n'
           '        "path": "src/a2a/client/base_client.py",\n        "start_line": 1,\n        "end_line": 100\n'
           '      }\n    }\n  ]\n}\n```')
    a = parse_action(raw)
    assert a["tool"] == "read_file" and a["args"]["path"] == "src/a2a/client/base_client.py"
