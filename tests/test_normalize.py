from normalize import files_from_text, funcs_from_text, norm_file

FILES = ["src/agents/run.py", "src/agents/models/litellm_model.py", "src/other/run.py", "src/agents/util.py"]
UNITS = {
    "src/agents/models/litellm_model.py": ["LitellmModel", "LitellmModel.get_response", "LitellmModel._fetch_response",
                                           "helper"],
    "src/agents/util.py": ["A.close", "B.close", "fmt"],
}


def test_files():
    assert norm_file("./src/agents/run.py", FILES) == "src/agents/run.py"
    assert norm_file("models/litellm_model.py", FILES) == "src/agents/models/litellm_model.py"
    assert norm_file("run.py", FILES) is None  # ambiguous suffix
    text = "1. src/agents/models/litellm_model.py\n2. agents/util.py\n3. nope.py\n4. src/agents/util.py"
    assert files_from_text(text, FILES) == ["src/agents/models/litellm_model.py", "src/agents/util.py"]


def test_functions():
    text = ("src/agents/models/litellm_model.py::get_response\n"
            "src/agents/models/litellm_model.py::LitellmModel._fetch_response\n"
            "src/agents/util.py::close\n"  # ambiguous: A.close and B.close
            "src/agents/util.py :: fmt\n"
            "src/agents/models/litellm_model.py::agents.models.litellm_model.LitellmModel.get_response\n")
    assert funcs_from_text(text, UNITS, FILES) == [
        "src/agents/models/litellm_model.py::LitellmModel.get_response",
        "src/agents/models/litellm_model.py::LitellmModel._fetch_response",
        "src/agents/util.py::fmt",
    ]


def test_pilot_findings():
    files = ["fastmcp_slim/fastmcp/client/telemetry.py", "fastmcp_slim/fastmcp/server/telemetry.py",
             "pkg/utils.py", "other/utils.py"]
    units = {"fastmcp_slim/fastmcp/client/telemetry.py": ["client_span", "Tracer.start"],
             "fastmcp_slim/fastmcp/server/telemetry.py": ["get_auth_span_attributes"]}
    # wrong root folder, unique trailing path
    assert norm_file("src/fastmcp/client/telemetry.py", files) == "fastmcp_slim/fastmcp/client/telemetry.py"
    # a bare basename is never enough
    assert norm_file("src/utils.py", files) is None
    # junk suffix and placeholder path with bare_ok
    assert funcs_from_text("fastmcp_slim/fastmcp/client/telemetry.py::client_span.def", units, files) == [
        "fastmcp_slim/fastmcp/client/telemetry.py::client_span"]
    assert funcs_from_text("path/to/file.py::Tracer.start", units, files, bare_ok=True) == [
        "fastmcp_slim/fastmcp/client/telemetry.py::Tracer.start"]
    assert funcs_from_text("path/to/file.py::Tracer.start", units, files) == []
    # bare qualname lines (G1 echoing a candidate without its path)
    assert funcs_from_text("Tracer.start\nget_auth_span_attributes", units, files, bare_ok=True) == [
        "fastmcp_slim/fastmcp/client/telemetry.py::Tracer.start",
        "fastmcp_slim/fastmcp/server/telemetry.py::get_auth_span_attributes"]
