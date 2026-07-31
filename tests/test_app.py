import asyncio

from humoid_tui.app import HumoidApp
from humoid_tui.screens import SettingsModelScreen


async def test_app_paints_while_services_initialize(monkeypatch):
    monkeypatch.setenv("HUMOID_GEMMA_AUTOSTART", "false")
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


async def test_settings_page_has_grouped_descriptive_controls(monkeypatch):
    monkeypatch.setenv("HUMOID_GEMMA_AUTOSTART", "false")

    async def initialize(_self):
        return None

    monkeypatch.setattr("humoid_tui.app.AgentHarness.initialize", initialize)
    app = HumoidApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(SettingsModelScreen())
        await pilot.pause()

        screen = app.screen
        assert screen.query_one("#active-provider").render()
        assert screen.query_one("#gemma-autostart-status").render()
        assert screen.query_one("#start-gemma").label == "START + USE GEMMA"
        assert len(screen.query(".settings-section")) == 5
