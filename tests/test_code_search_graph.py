from __future__ import annotations

from dataclasses import dataclass, field

from humoid_tui.config import Settings
from humoid_tui.perspective import ProjectPerspectiveIndex


@dataclass
class FakeMemoryHit:
    text: str
    score: float = 1.0
    channel: str = "conversation"
    validation_status: str = "verified"
    metadata: dict[str, object] = field(default_factory=dict)


class FakeFailureMemory:
    status = "weaviate: healthy test double"

    async def initialize(self) -> None:
        return

    async def search(self, query: str, limit: int = 8) -> list[FakeMemoryHit]:
        assert "regression" in query
        return [
            FakeMemoryHit(
                text=(
                    "Previous auth regression touched src/auth/session.py. "
                    "SessionManager.refresh caused expired-token retry failures."
                ),
                metadata={
                    "kind": "bug",
                    "files": ["src/auth/session.py"],
                    "symbols": ["SessionManager", "refresh"],
                },
            )
        ][:limit]

    async def close(self) -> None:
        return


def local_settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("HUMOID_PERSPECTIVE_BACKEND", "local")
    monkeypatch.setenv("HUMOID_PERSPECTIVE_CHUNK_CHARS", "512")
    monkeypatch.setenv("HUMOID_PERSPECTIVE_CHUNK_OVERLAP", "64")
    monkeypatch.setenv("HUMOID_PERSPECTIVE_PACKET_MAX_CHARS", "5000")
    return Settings(humoid_workspace=tmp_path, humoid_memory_backend="sqlite")


async def test_constraint_graph_returns_minimal_connected_behavior_packet(
    tmp_path,
    monkeypatch,
):
    settings = local_settings(tmp_path, monkeypatch)
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "auth" / "session.py").write_text(
        "class SessionManager:\n"
        "    def refresh(self, token):\n"
        "        return rotate_token(token)\n\n"
        "def rotate_token(token):\n"
        "    return token + '-new'\n"
    )
    (tmp_path / "src" / "api.py").write_text(
        "from auth.session import SessionManager\n\n"
        "def send(request, session):\n"
        "    if request.status == 401:\n"
        "        session.refresh(request.token)\n"
        "    return request.retry()\n"
    )
    (tmp_path / "tests" / "test_session.py").write_text(
        "from src.auth.session import SessionManager\n\n"
        "def test_expired_token_refresh():\n"
        "    manager = SessionManager()\n"
        "    assert manager.refresh('old')\n"
    )
    (tmp_path / "src" / "unrelated.py").write_text(
        "def refresh_ui():\n"
        "    return 'refresh token label'\n"
    )

    perspective = ProjectPerspectiveIndex(settings, tmp_path)
    built = await perspective.build("locate expired token refresh behavior")
    assert "constraint-graph" in built
    assert perspective.manifest["graph_edges"] >= 2

    packet = await perspective.search(
        "Where is SessionManager.refresh implemented and tested?",
        limit=5,
    )

    assert "minimal connected behavior subgraph" in packet
    assert "src/auth/session.py" in packet
    assert "tests/test_session.py" in packet
    assert "BEHAVIOR GRAPH EDGES" in packet
    assert "symbol:sessionmanager" in packet
    assert "src/unrelated.py" not in packet.split("[CONTEXT ACCORDION", 1)[0]


async def test_failure_memory_boosts_historical_bug_zone(tmp_path, monkeypatch):
    settings = local_settings(tmp_path, monkeypatch)
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "session.py").write_text(
        "class SessionManager:\n"
        "    def refresh(self, token):\n"
        "        return token\n"
    )
    (tmp_path / "src" / "auth" / "alternate.py").write_text(
        "class AlternateSession:\n"
        "    def refresh(self, token):\n"
        "        return token\n"
    )

    perspective = ProjectPerspectiveIndex(
        settings,
        tmp_path,
        memory=FakeFailureMemory(),
    )
    await perspective.build("fix the expired-token regression")
    packet = await perspective.search("debug token refresh regression", limit=2)

    assert "FAILURE MEMORY" in packet
    assert "failure_memory=weaviate: healthy test double" in packet
    assert "past failure touched src/auth/session.py" in packet
    primary = packet.split("[CONTEXT ACCORDION", 1)[0]
    assert primary.index("src/auth/session.py") < primary.index("src/auth/alternate.py")
