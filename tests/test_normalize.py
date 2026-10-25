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
