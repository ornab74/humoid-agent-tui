import pytest

from humoid_tui.action_guard import deferred_action, mutation_requested, _guarded_run
from humoid_tui.models import AgentEvent


class DummyMemory:
    def __init__(self):
        self.rows = []

    async def add(self, text, **meta):
        self.rows.append((text, meta))
        return "memory-1"


class DummyTools:
    def __init__(self):
        self.undo_stack = []


class DummyProvider:
    class Config:
        name = "digitalocean"
        model = "glm-test"

    cfg = Config()


class DummySettings:
    humoid_tool_retry_limit = 2


class DummyProtocol:
    name = "glm-qwen-adaptive"


class DummyHarness:
    def __init__(self):
        self.events = []
        self.memory = DummyMemory()
        self.tools = DummyTools()
        self.provider = DummyProvider()
        self.s = DummySettings()
        self.messages = []
        self.protocol = DummyProtocol()

        async def emit(event):
            self.events.append(event)

        self.emit = emit


@pytest.mark.parametrize(
    "prompt",
    [
        "append these ideas to ideas.md",
        "edit the README file and update its table",
        "modify the project code",
    ],
)
def test_mutation_request_detection(prompt):
    assert mutation_requested(prompt)


def test_deferred_action_detection_matches_old_failure_language():
    assert deferred_action("Let me write the complete file with additions.")
    assert deferred_action("I need to construct the full updated content and write it now.")
    assert not deferred_action("Updated ideas.md successfully.")


@pytest.mark.asyncio
async def test_protocol_errors_abort_early_and_force_action(monkeypatch):
    monkeypatch.setenv("HUMOID_ACTION_GUARD_ATTEMPTS", "2")
    monkeypatch.setenv("HUMOID_ACTION_GUARD_PROTOCOL_ERRORS", "2")
    harness = DummyHarness()
    calls = 0

    async def stalled_run(active_harness, prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            await active_harness.emit(AgentEvent("protocol_error", "invalid tool JSON"))
            await active_harness.emit(AgentEvent("protocol_error", "invalid tool JSON again"))
            pytest.fail("the action guard should interrupt before a third round")
        active_harness.tools.undo_stack.append(("ideas.md", None))
        await active_harness.emit(AgentEvent("token", "Updated ideas.md."))
        return "Updated ideas.md."

    result = await _guarded_run(harness, stalled_run, "append ideas to ideas.md")

    assert calls == 2
    assert result == "Updated ideas.md."
    assert any(event.kind == "action_guard" for event in harness.events)
    assert len(harness.memory.rows) == 1


@pytest.mark.asyncio
async def test_planning_only_result_gets_one_action_retry(monkeypatch):
    monkeypatch.setenv("HUMOID_ACTION_GUARD_ATTEMPTS", "2")
    harness = DummyHarness()
    prompts = []

    async def staged_run(active_harness, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return "Let me carefully construct the full file and write it now."
        active_harness.tools.undo_stack.append(("notes.md", None))
        return "Updated notes.md."

    result = await _guarded_run(harness, staged_run, "insert a section into notes.md")

    assert result == "Updated notes.md."
    assert len(prompts) == 2
    assert prompts[1].startswith("ACTION GUARD:")
    assert len(harness.memory.rows) == 1
