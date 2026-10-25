"""Method G2: graph agent with the competition's tool signatures (text-JSON protocol).

Each turn the model replies with ONE JSON object:
  {"tool": "get_code_neighbors", "args": {"node": "...", "edge_type": null, "max_neighbors": 50}}
  {"tool": "search_similar_code", "args": {"query": "...", "k": 10}}
  {"tool": "get_code_subgraph", "args": {"nodes": ["...", "..."]}}
  {"tool": "read_file", "args": {"path": "...", "start_line": 1, "end_line": 80}}
  {"final": ["path/to/file.py::Class.method", ...]}          (up to 10, most likely first)

The agent sees the issue and the core-file layout (sorted, capped), not the BM25
ranking. search_similar_code embeds the query with EmbeddingGemma 2 when
--embed is given (node vectors share the B2 content-hash cache), else BM25 over
node text. Tool-call validity is logged per step (PREREG D2).

  python -I loc/run_g2.py --model gemma4:e4b-it-qat --tag e4b --ids-file ... --backend ollama \
      --base-url http://127.0.0.1:11434 [--embed google/embeddinggemma-2]
Do not run on evaluation ids before PREREG.md is committed (pilot ids only).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "graph"))
sys.path.insert(0, str(ROOT / "harvest"))
import snapshots as S  # noqa: E402
from build_graph import build  # noqa: E402
from data import V01, load, read_jsonl  # noqa: E402
from llm import chat  # noqa: E402
from normalize import files_from_text, funcs_from_text  # noqa: E402
from paths import NON_CORE_RE, is_core_py  # noqa: E402
from tools import CodeGraph  # noqa: E402

SYSTEM = """You localize the code that must change to resolve a GitHub issue in a Python repository.
Reply with exactly ONE JSON object per turn and nothing else.
Tools (node ids are dotted symbols such as pkg.module.Class.method; short suffixes like Class.method also work):
- {"tool": "search_similar_code", "args": {"query": "<text or symbol>", "k": 10}}
- {"tool": "get_code_neighbors", "args": {"node": "<symbol>", "edge_type": null, "max_neighbors": 30}}
- {"tool": "get_code_subgraph", "args": {"nodes": ["<symbol>", "<symbol>"]}}
- {"tool": "read_file", "args": {"path": "<file.py>", "start_line": 1, "end_line": 80}}
When you are confident, answer with {"final": ["path/to/file.py::Class.method", "path/to/file.py::function"]}
listing up to 10 functions or methods that must be edited, most likely first.
Your first reply must be a tool call: use at least {min_tools} tool calls before the final answer, and at most {steps}."""
GRAPH_CACHE: dict = {}
JSON_RE = re.compile(r"\{.*\}", re.S)


def blob_map(root: Path) -> dict[str, str]:
    p = subprocess.run(["git", "-c", "core.quotepath=false", "ls-tree", "-r", "HEAD"], cwd=root,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return {ln.partition("\t")[2]: ln.split()[2] for ln in p.stdout.splitlines() if ln.endswith(".py")}


TOOL_KEYS = ("tool", "name", "tool_name", "function")
ARG_KEYS = ("args", "arguments", "parameters", "input")


def _canonical(obj: dict) -> dict:
    """Accept common function-calling shapes ({"name":..,"arguments":..}, nested "function")."""
    if "final" in obj:
        return obj
    calls = obj.get("tool_calls")
    if isinstance(calls, list) and calls and isinstance(calls[0], dict):
        return _canonical(calls[0])  # pilot (E2B): OpenAI-style list; one tool per turn, take the first
    fn = obj.get("function")
    if isinstance(fn, dict):
        obj = {**obj, **fn}
    tool = next((obj[k] for k in TOOL_KEYS if isinstance(obj.get(k), str)), None)
    args = next((obj[k] for k in ARG_KEYS if k in obj), {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {"tool": tool, "args": args if isinstance(args, dict) else {}} if tool else obj


def parse_action(text: str) -> dict | None:
    m = JSON_RE.search(text or "")
    if not m:
        return None
    s = m.group(0)
    for end in range(len(s), 0, -1):  # tolerate trailing chatter after the object
        if s[end - 1] != "}":
            continue
        try:
            obj = json.loads(s[:end])
            return _canonical(obj) if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def run_one(r: dict, args, encoder) -> dict:
    root = S.checkout(r["repo"], r["base_commit"])
    blobs = blob_map(root)
    files = sorted(f for f in blobs if not NON_CORE_RE.search(f) and (root / f).exists())
    g = build(root, files, r["repo"], r["base_commit"], cache=GRAPH_CACHE, blob_of=blobs)
    core = [f for f in files if is_core_py(f)]
    mod_of = {n["file"]: n["id"] for n in g["nodes"] if n["type"] == "module"}
    emb = qenc = None
    if encoder is not None:
        syms = [n for n in g["nodes"] if n["type"] != "module" and is_core_py(n["file"])]
        docs = [encoder.doc(f"{n['file']} {n['id'][len(mod_of[n['file']]) + 1:]}", n["text"]) for n in syms]
        vecs = encoder.embed(docs)
        emb = {n["id"]: v for n, v in zip(syms, vecs)}
        qenc = encoder.query
    cg = CodeGraph(g, embeddings=emb, query_encoder=qenc)
    layout = "\n".join(core[:200]) + (f"\n... ({len(core) - 200} more files)" if len(core) > 200 else "")
    system = SYSTEM.replace("{steps}", str(args.steps)).replace("{min_tools}", str(args.min_tools))
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Repository: {r['repo']}\nIssue:\n{r['problem_statement'][:6000]}\n\n"
                                        f"Core Python files:\n{layout}"}]
    log, calls, final = [], [], None
    for step in range(args.steps + 1):
        c = chat(msgs, model=args.model, base_url=args.base_url, max_tokens=400)
        calls.append(c)
        act = parse_action(c["text"])
        msgs.append({"role": "assistant", "content": c["text"][:2000]})
        n_tools = sum(1 for x in log if "tool" in x)
        if act and "final" in act and (n_tools >= args.min_tools or step == args.steps):
            final = act["final"] if isinstance(act["final"], list) else []
            log.append({"step": step, "final": True})
            break
        if act and "final" in act:  # pilot finding: both models answered before exploring
            log.append({"step": step, "early_final": True})
            msgs.append({"role": "user", "content": f"Use the tools first: make at least {args.min_tools} tool "
                                                    "calls before the final answer. Reply with a tool call now."})
            continue
        if step == args.steps:
            break
        valid, result = False, ""
        if act and act.get("tool") in ("get_code_neighbors", "search_similar_code", "get_code_subgraph", "read_file"):
            a = act.get("args") or {}
            try:
                if act["tool"] == "get_code_neighbors":
                    result = cg.get_code_neighbors(str(a.get("node", "")), a.get("edge_type"), int(a.get("max_neighbors", 30)))
                elif act["tool"] == "search_similar_code":
                    result = cg.search_similar_code(str(a.get("query", "")), int(a.get("k", 10)))
                elif act["tool"] == "get_code_subgraph":
                    nodes = a.get("nodes") or []
                    result = cg.get_code_subgraph([str(x) for x in (nodes if isinstance(nodes, list) else [nodes])][:30])
                else:
                    path = str(a.get("path", "")).lstrip("./")
                    path = path if path in blobs else next((f for f in files if f.endswith("/" + path)), path)
                    lines = (root / path).read_text(encoding="utf-8", errors="replace").splitlines()
                    s0 = max(1, int(a.get("start_line", 1)))
                    s1 = min(len(lines), int(a.get("end_line", s0 + 79)), s0 + 149)
                    result = "\n".join(f"{i}: {lines[i - 1]}" for i in range(s0, s1 + 1))
                valid = True
            except Exception as e:  # tool error is returned to the model, like the harness does
                result = json.dumps({"status": "error", "error": f"{type(e).__name__}: {e}"[:300]})
        else:
            result = ('Invalid reply. Answer with ONE JSON object: a tool call {"tool": ..., "args": {...}} '
                      'or {"final": [...]}')
        entry = {"step": step, "tool": (act or {}).get("tool"), "valid": valid}
        if not valid:
            entry["raw"] = (c["text"] or "")[:200]  # diagnosis of invalid replies
        log.append(entry)
        msgs.append({"role": "user", "content": f"Tool result:\n{result[:3000]}"})
    # map final answers (paths or dotted ids) onto canonical path::qual keys
    id_to_key = {}
    for n in g["nodes"]:
        if n["type"] != "module" and n["file"] in mod_of:
            id_to_key[n["id"]] = f"{n['file']}::{n['id'][len(mod_of[n['file']]) + 1:]}"
    units = {}
    for k in id_to_key.values():
        f, _, q = k.partition("::")
        units.setdefault(f, []).append(q)
    funcs = []
    for item in final or []:
        item = str(item)
        if "::" in item:
            keys = funcs_from_text(item, units, files, bare_ok=True)
        elif cg.resolve(item) in id_to_key:
            keys = [id_to_key[cg.resolve(item)]]
        else:
            keys = funcs_from_text(item, units, files, bare_ok=True)
        funcs += [k for k in keys if k not in funcs]
    files_ranked = list(dict.fromkeys(k.partition("::")[0] for k in funcs))
    files_ranked += [f for f in files_from_text(" ".join(str(x) for x in final or []), core)
                     if f not in files_ranked]  # file-only answers still rank files
    return {"files": files_ranked, "functions": funcs[:10], "steps": log, "calls": calls,
            "finished": final is not None, "raw_final": str(final)[:400]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--backend", choices=["openai", "ollama"], default=None)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--min-tools", type=int, default=2)
    ap.add_argument("--embed", default="", help="HF id of the query/node encoder (e.g. google/embeddinggemma-2)")
    ap.add_argument("--embed-device", default="cuda", help="e.g. cuda:1 to keep the encoder off the LLM's GPU")
    args = ap.parse_args()
    if args.backend:
        import os
        os.environ["AGENTLOC_LLM_BACKEND"] = args.backend
    encoder = None
    if args.embed:
        from dense import STEncoder
        encoder = STEncoder("embeddinggemma-2", args.embed, device=args.embed_device)
    out = V01 / "runs" / f"G2-{args.tag}.jsonl"
    done = {x["instance_id"] for x in read_jsonl(out)} if out.exists() else set()
    keep = {x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
    insts = sorted((r for r in load() if r["instance_id"] in keep and r["instance_id"] not in done),
                   key=lambda r: (r["repo"], r["merged_at"]))
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(insts, 1):
            try:
                res = run_one(r, args, encoder)
            except Exception as e:  # record an empty prediction, never skip silently
                res = {"files": [], "functions": [], "steps": [], "calls": [], "finished": False,
                       "error": f"{type(e).__name__}: {e}"[:300]}
            calls = res.pop("calls")
            res.update(instance_id=r["instance_id"], model=args.model,
                       prompt_tokens=sum(c.get("prompt_tokens") or 0 for c in calls),
                       completion_tokens=sum(c.get("completion_tokens") or 0 for c in calls),
                       seconds=round(sum(c.get("seconds") or 0 for c in calls), 2), n_calls=len(calls),
                       tool_calls=sum(1 for s in res["steps"] if s.get("tool")),
                       invalid_replies=sum(1 for s in res["steps"] if "tool" in s and not s.get("valid")),
                       valid_tool_calls=sum(1 for s in res["steps"] if s.get("valid")))
            fh.write(json.dumps(res) + chr(10))
            fh.flush()
            if n % 5 == 0:
                print(f"  {n}/{len(insts)}, {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(insts)} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
