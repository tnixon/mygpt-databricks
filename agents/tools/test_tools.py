"""Tests for the Phase-5 UC-function tools (agents/tools/sql/02_functions.sql).

    uv run --no-project --with-requirements eval/requirements.txt pytest -q agents/tools/test_tools.py

The Python UDF bodies are executed straight out of the SQL file, so the test covers the code
Unity Catalog runs. The SQL-language functions (RCSB / NCBI) were checked against recorded API
payloads on a warehouse; see PLAN.md § Phase 5 spikes. Here only their contract is checked.
"""

from __future__ import annotations

import os
import re

SQL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql", "02_functions.sql")
# Functions are created unqualified under USE SCHEMA :tools_schema, or (when the body reads the
# data schema) as IDENTIFIER(:tools_schema || '.<fn>').
FN_NAME = r"(?:IDENTIFIER\(:tools_schema \|\| '\.(?P<q>\w+)'\)|(?P<u>\w+))"


def _sql() -> str:
    with open(SQL, encoding="utf-8") as f:
        return f.read()


def python_udf(name: str):
    """Compile a LANGUAGE PYTHON function body from the SQL file into a callable."""
    m = re.search(
        rf"FUNCTION\s+(?:IDENTIFIER\(:tools_schema \|\| '\.{name}'\)|{name})\((?P<params>.*?)\)\s*RETURNS.*?AS \$\$\n(?P<body>.*?)\$\$",
        _sql(), re.S,
    )
    assert m, f"{name}: no Python body"
    params = [p.split()[0] for p in re.findall(r"^\s*(\w+\s+\w+)", m.group("params"), re.M)]
    body = "\n".join("    " + line for line in m.group("body").splitlines())
    ns: dict = {}
    exec(f"def {name}({', '.join(params)}):\n{body}\n", ns)  # noqa: S102  (our own file)
    return ns[name]


def test_pubmed_ids_from_text():
    f = python_udf("pubmed_ids_from_text")
    assert f("What does PMID 10498895 say about EphA8?") == ["10498895"]
    # bare 7- and 8-digit ids, order kept, duplicates dropped
    assert f("compare 10838079 and 9876543, then 10838079 again") == ["10838079", "9876543"]
    # explicit mentions accept short ids; URLs count
    assert f("PMID: 123 and https://pubmed.ncbi.nlm.nih.gov/17164530/") == ["123", "17164530"]
    assert f("PubMed ID 00012345") == ["12345"]
    # no false hits on years, short numbers or long accession-like numbers
    assert f("In 2007, 42 patients; NCT0123456789") == []
    assert f("") == [] and f(None) == []


def test_every_tool_is_documented():
    """MCP tool descriptions and argument schemas come from the COMMENTs."""
    sql = _sql()
    fns = [(m["q"] or m["u"], m["params"], m["tail"]) for m in re.finditer(
        rf"CREATE OR REPLACE FUNCTION {FN_NAME}\((?P<params>.*?)\)\s*RETURNS(?P<tail>.*?)(?:AS \$\$|RETURN)", sql, re.S)]
    names = {n for n, _, _ in fns}
    assert names == {"pubmed_ids_from_text", "pubmed_summary", "search_pdb_by_gene", "search_pdb",
                     "get_pdb_details", "list_corpora", "list_corpus_documents"}
    for name, params, tail in fns:
        assert "COMMENT '" in tail, f"{name}: missing function COMMENT"
        for param in filter(str.strip, re.split(r",\s*\n", params)):
            assert "COMMENT '" in param, f"{name}: parameter without COMMENT: {param.strip()[:40]}"


def test_external_calls_use_governed_connections():
    sql = _sql()
    conns = set(re.findall(r"http_request\(conn => '(\w+)'", sql))
    assert conns == {"rcsb_search", "rcsb_data", "ncbi_eutils"}
    with open(os.path.join(os.path.dirname(SQL), "01_connections.sql"), encoding="utf-8") as f:
        declared = set(re.findall(r"CREATE CONNECTION IF NOT EXISTS (\w+)", f.read()))
    assert conns <= declared


def test_sql_parentheses_balance():
    """The SQL functions are deployed unexecuted; an unbalanced paren only surfaces at deploy."""
    for stmt in re.split(r";\s*\n", _sql()):
        s = re.sub(r"--[^\n]*", "", stmt)
        s = re.sub(r"\$\$.*?\$\$", "", s, flags=re.S)
        s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", s)
        depth = 0
        for ch in s:
            depth += (ch == "(") - (ch == ")")
            assert depth >= 0, f"unbalanced ')' in: {s.strip()[:80]}"
        assert depth == 0, f"unclosed '(' in: {s.strip()[:80]}"
