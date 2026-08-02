from __future__ import annotations

from humoid_tui.code_intelligence import CodeIntelligenceIndex


def test_incremental_python_ast_graph_and_cache(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "class BaseService:\n"
        "    def execute(self, value):\n"
        "        return value\n\n"
        "class PaymentService(BaseService):\n"
        "    def execute(self, value):\n"
        "        return validate_payment(value)\n\n"
        "def validate_payment(value):\n"
        "    return bool(value)\n"
    )
    (tmp_path / "src" / "api.py").write_text(
        "from src.service import PaymentService\n\n"
        "def charge(value):\n"
        "    return PaymentService().execute(value)\n"
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "from src.service import PaymentService\n\n"
        "def test_payment_execute():\n"
        "    assert PaymentService().execute('ok')\n"
    )

    index = CodeIntelligenceIndex(tmp_path)
    first = index.build()
    assert first["parsed"] == 3
    assert first["symbols"] >= 7
    assert any(edge.kind == "inherits" for edge in index.edges)
    assert any(edge.kind == "calls" and "validate_payment" in edge.evidence for edge in index.edges)
    assert any(edge.kind == "tests" for edge in index.edges)

    second = CodeIntelligenceIndex(tmp_path).build()
    assert second["parsed"] == 0
    assert second["reused"] == 3


def test_impact_frontier_finds_callers_subclasses_and_tests(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "session.py").write_text(
        "class SessionBase:\n"
        "    def refresh(self, token):\n"
        "        return token\n\n"
        "class SessionManager(SessionBase):\n"
        "    def refresh(self, token):\n"
        "        return rotate_token(token)\n\n"
        "def rotate_token(token):\n"
        "    return token + '-new'\n"
    )
    (tmp_path / "src" / "client.py").write_text(
        "from src.session import SessionManager\n\n"
        "def retry(token):\n"
        "    return SessionManager().refresh(token)\n"
    )
    (tmp_path / "tests" / "test_session.py").write_text(
        "from src.session import SessionManager\n\n"
        "def test_refresh():\n"
        "    assert SessionManager().refresh('old')\n"
    )

    index = CodeIntelligenceIndex(tmp_path)
    index.build()
    report = index.impact("fix SessionManager.refresh token regression", max_depth=4, max_nodes=30)
    rendered = report.render()

    assert "STATIC IMPACT ANALYSIS" in rendered
    assert "src/session.py" in report.files
    assert "src/client.py" in report.files
    assert "tests/test_session.py" in report.tests
    assert any(edge.kind in {"calls", "inherits", "tests"} for edge in report.edges)


def test_localization_benchmark_reports_metrics(tmp_path):
    (tmp_path / "auth.py").write_text(
        "class TokenStore:\n"
        "    def load(self):\n"
        "        return read_token()\n\n"
        "def read_token():\n"
        "    return 'token'\n"
    )
    index = CodeIntelligenceIndex(tmp_path)
    index.build()
    result = index.benchmark([
        {"query": "where does TokenStore load the token", "expected_files": ["auth.py"]}
    ])
    assert result["cases"] == 1
    assert result["mean_recall"] == 1.0
    assert result["mean_reciprocal_rank"] == 1.0


def test_main_installs_all_runtime_layers():
    source = __import__("pathlib").Path("src/humoid_tui/__main__.py").read_text()
    assert source.index("install_edit_cycle_runtime()") < source.index("install_code_intelligence_runtime()")
    assert source.index("install_code_intelligence_runtime()") < source.index("install_action_guard()")
