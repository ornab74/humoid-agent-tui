import asyncio

from humoid_tui.app import HumoidApp


async def test_app_paints_while_services_initialize(monkeypatch):
    initialization_started = asyncio.Event()
    allow_initialization = asyncio.Event()

    async def slow_initialize(_self):
        initialization_started.set()
        await allow_initialization.wait()

    monkeypatch.setattr(
        "humoid_tui.app.AgentHarness.initialize",
        slow_initialize,
    )

    app = HumoidApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await asyncio.wait_for(initialization_started.wait(), timeout=1)

        assert app.query_one("#topbar").region.height == 3
        assert app.query_one("#main").region.height > 0
        # Local commands and cancellation remain available during startup.
        assert not app.query_one("#command").disabled

        allow_initialization.set()
        await pilot.pause()

        assert not app.query_one("#command").disabled
