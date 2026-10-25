from mask import mask_issue


def test_mask_keeps_prose_and_hides_code():
    text = ("When I call `Runner.run` the tracing exporter crashes in src/agents/tracing/processors.py.\n"
            "The function _truncate_json_value fails for LitellmModel and agents.run.Runner, see HTTP 500.\n"
            "```python\nprint('x')\n```\nThe processor never retries.")
    out = mask_issue(text, symbols={"processor", "retries", "run"})
    assert "Runner.run" not in out and "processors.py" not in out and "_truncate_json_value" not in out
    assert "LitellmModel" not in out and "agents.run.Runner" not in out and "print(" not in out
    assert "tracing exporter crashes" in out and "HTTP 500" in out
    assert "[SYMBOL] never [SYMBOL]" in out  # repository symbols >= 4 chars are masked
    assert "call" in out  # ordinary words stay
