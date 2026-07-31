from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Input, RichLog, Static

from .agent import AgentHarness
from .config import ProviderName, Settings
from .context_accordion import ContextAccordion
from .env_store import EnvStore
from .i18n import LOCALES, translate
from .local_models import LocalModelManager
from .models import AgentEvent
from .preferences import PreferenceStore
from .screens import (
    ContextAccordionScreen,
    DiffApprovalScreen,
    MemoryBrowserScreen,
    ReferenceScreen,
    LanguageScreen,
    SessionsScreen,
    SettingsModelScreen,
)
from .telemetry import SessionTelemetry

CSS = """
Screen { background: #03080d; color: #c4d3dd; }
#topbar { height: 3; border: solid #263744; background: #050b12; padding: 0 1; color: #d7e1e8; }
#session-tabs { height: 2; background: #07121a; color: #a78bfa; padding: 0 1; }
#main { height: 1fr; }
#left { width: 28%; min-width: 38; }
.panel { border: solid #263744; background: #050b12; }
.panel-title { height: 2; padding: 0 1; color: #10d9ed; text-style: bold; border-bottom: solid #263744; }
#activity { height: 50%; }
#agents { height: 25%; padding: 0 1; }
#session { height: 25%; padding: 0 1; }
#workspace { width: 72%; }
#workspace-title { height: 3; padding: 0 1; color: #10d9ed; border-bottom: solid #263744; }
#chat { height: 1fr; padding: 0 1; scrollbar-color: #00aebd; }
#metrics { height: 3; border: solid #263744; background: #050b12; padding: 0 1; color: #b8c7d1; }
#command { height: 3; border: solid #263744; background: #050b12; padding: 0 1; }
#keys { height: 2; border: solid #263744; background: #050b12; padding: 0 1; color: #a78bfa; }
#bottom-nav { height: 5; border: solid #263744; background: #050b12; align: center middle; }
#bottom-nav Button { width: 18; min-width: 11; height: 3; margin: 0 1; color: #d7e1e8; }
#bottom-nav Button:focus { background: #10d9ed; color: #03080d; text-style: bold; }
RichLog { scrollbar-color: #00aebd; }
Button:focus { background: #10d9ed; color: #03080d; text-style: bold; }
Input:focus, DataTable:focus, TextArea:focus { border: tall #10d9ed; }
"""

class HumoidApp(App):
    CSS = CSS
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_logs", "Clear"),
        Binding("ctrl+1", "help_reference", "Help", priority=True),
        Binding("ctrl+2", "memory", "Memory", priority=True),
        Binding("ctrl+3", "context", "Context", priority=True),
        Binding("ctrl+4", "model_settings", "Settings", priority=True),
        Binding("down", "focus_navigation", "Menu", priority=True),
        Binding("up", "leave_navigation", "Input", priority=True),
        Binding("left", "navigation_left", "Previous", priority=True),
        Binding("right", "navigation_right", "Next", priority=True),
        Binding("f1", "help_reference", "Help"),
        Binding("f2", "agents", "Agents"),
        Binding("f3", "gemma_menu", "Gemma 4"),
        Binding("f4", "tools", "Tools"),
        Binding("f5", "memory", "Memory"),
        Binding("f6", "context", "Context"),
        Binding("f9", "settings", "Settings"),
        Binding("f12", "model_settings", "Models"),
        Binding("escape", "cancel_active", "Cancel", priority=True),
        Binding("ctrl+c", "cancel_active", "Cancel", priority=True),
    ]

    def __init__(self):
        super().__init__()
        self.env_store = EnvStore()
        self.preference_store = PreferenceStore()
        saved = {
            key.lower(): value
            for key, value in self.preference_store.values().items()
            if key.lower() in Settings.model_fields
        }
        self.settings = Settings(**saved)
        self.harness: AgentHarness | None = None
        self.session = uuid.uuid4().hex[:8]
        self.telemetry = SessionTelemetry(context_limit=self.settings.context_limit())
        self._stream_open = False
        self.harness_ready = False
        self.active_worker = None
        self.model_manager = LocalModelManager()
        self.language = self.settings.humoid_language if self.settings.humoid_language in LOCALES else "en"
        self.sessions: dict[str, dict[str, object]] = {}

    def compose(self) -> ComposeResult:
        yield Static("Humoid Agent TUI", id="topbar")
        yield Static("", id="session-tabs")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                with Vertical(classes="panel", id="activity"):
                    yield Static("ACTIVITY STREAM                         filter: all", classes="panel-title", id="activity-title")
                    yield RichLog(id="activity-log", wrap=True, highlight=True, markup=True)
                with Vertical(classes="panel", id="agents"):
                    yield Static("AGENT TREE", classes="panel-title", id="agents-title")
                    yield Static("🌱 root (you)\n ├─ ● planner\n │  ├─ ◉ file_analyst [idle]\n │  └─ ● code_reader\n ├─ ● tool_router\n └─ ● memory_agent", id="agent-tree")
                with Vertical(classes="panel", id="session"):
                    yield Static("SESSION INFO", classes="panel-title", id="session-title")
                    yield Static("Starting…", id="session-info")
            with Vertical(classes="panel", id="workspace"):
                yield Static("▣  CONVERSATION & WORKSPACE", id="workspace-title")
                yield RichLog(id="chat", wrap=True, highlight=True, markup=True)
        yield Static("CTX 0%   TOK 0   TOOLS 0   MEM …   TIME 00:00   MODE booting", id="metrics")
        yield Input(placeholder="› Type your message or command…", id="command")
        with Horizontal(id="bottom-nav"):
            yield Button("LANGUAGE", id="nav-language")
            yield Button("HELP", id="nav-help")
            yield Button("SESSIONS", id="nav-sessions")
            yield Button("MEMORY", id="nav-memory")
            yield Button("CONTEXT", id="nav-context")
            yield Button("SETTINGS", id="nav-settings")

    async def on_mount(self):
        self.harness = AgentHarness(self.settings, self.emit_event)
        self.harness.tools.approve_write = self.approve_file_change
        self.sessions[self.session] = {
            "title": "Session 1", "messages": self.harness.messages,
            "accordion": self.harness.context_accordion,
        }
        command = self.query_one("#command", Input)
        self._refresh_header()
        self._refresh_metrics()
        command.focus()
        self._apply_language()
        self._refresh_session_tabs()
        self.run_worker(
            self._initialize_harness(),
            name="initialize-harness",
            exclusive=True,
        )

    async def _initialize_harness(self) -> None:
        """Initialize slow services without delaying the first paint."""

        try:
            await self.harness.initialize()
            self.harness_ready = True
            if (
                self.settings.humoid_gemma_autostart
                and self.settings.humoid_provider == "llamacpp"
            ):
                await self.ensure_local_gemma()
        except Exception as exc:
            await self.emit_event(AgentEvent("error", str(exc)))
        finally:
            try:
                self._refresh_header()
                self._refresh_metrics()
                command = self.query_one("#command", Input)
                command.focus()
            except NoMatches:
                # The app may be shutting down while startup is cancelled.
                pass

    async def approve_file_change(self, path: str, diff: str, content: str) -> tuple[bool, str]:
        return await self.push_screen_wait(DiffApprovalScreen(path, diff, content))

    async def switch_active_provider(self, name: ProviderName, *, persist: bool = True) -> None:
        """Switch the live harness and optionally remember the selection."""
        if not self.harness:
            raise RuntimeError("Agent services are not initialized")
        self.settings.humoid_provider = name
        if persist:
            self.save_preference("HUMOID_PROVIDER", name)
        await self.harness.switch_provider(name)
        context_limit = self.settings.context_limit(name)
        self.telemetry.context_limit = context_limit
        for state in self.sessions.values():
            state["accordion"].context_limit = context_limit
        self.harness.context_accordion.context_limit = context_limit
        self._refresh_header()
        self._refresh_metrics()

    def save_preference(self, key: str, value: object) -> None:
        """Persist a user setting to SQLite and mirror it to .env."""
        self.preference_store.set(key, value)
        self.env_store.set(key, value)

    def _configured_gemma_path(self) -> Path:
        """Resolve LLAMACPP_MODEL to a downloaded GGUF without guessing silently."""
        configured = Path(self.settings.llamacpp_model).name
        candidates = self.model_manager.models()
        for path in candidates:
            if configured in {path.name, path.stem}:
                return path
        if len(candidates) == 1:
            return candidates[0]
        raise RuntimeError(
            f"Configured local model {configured!r} was not found in "
            f"{self.model_manager.model_dir}. Select a GGUF in Settings."
        )

    async def ensure_local_gemma(self) -> str:
        """Idempotently start the managed server and activate its provider."""
        path = self._configured_gemma_path()
        manager = self.model_manager
        if not manager.process or manager.process.returncode is not None:
            await self.emit_event(AgentEvent("provider", f"Starting local Gemma: {path.name}"))
            await manager.launch(
                path,
                context=self.settings.humoid_gemma_context_limit,
            )
        await manager.wait_until_ready()
        self.settings.llamacpp_model = path.stem
        self.settings.llamacpp_base_url = "http://127.0.0.1:8080/v1"
        self.save_preference("LLAMACPP_MODEL", path.stem)
        self.save_preference("LLAMACPP_BASE_URL", self.settings.llamacpp_base_url)
        if (
            self.harness.provider.cfg.name != "llamacpp"
            or self.harness.provider.cfg.model != path.stem
        ):
            await self.switch_active_provider("llamacpp")
        return f"Local Gemma ready: {path.name}"

    def _refresh_header(self):
        p = self.harness.provider.cfg if self.harness else self.settings.provider()
        protocol = self.harness.protocol.name if self.harness else "booting"
        gemma = self.harness.gemma4 if self.harness else None
        self.query_one("#topbar", Static).update(
            f"🤖 Humoid Agent TUI v0.3.0     Agent: root     Model: {p.model or 'unset'}     "
            f"Provider: {p.name}     Protocol: {protocol}     Session: {self.session}     ● CONNECTED"
        )
        self.query_one("#workspace-title", Static).update(
            f"▣  {self.tr('workspace')}                         "
            f"Gemma 4: {(gemma.mode.upper() if gemma else 'OFF')}    Thinking: {(gemma.thinking.upper() if gemma else 'OFF')}"
        )

    def _refresh_metrics(self):
        t = self.telemetry
        provider = self.harness.provider.cfg.name if self.harness else "booting"
        proto = self.harness.protocol.name if self.harness else "booting"
        self.query_one("#metrics", Static).update(
            f"▣ CTX {t.context_percent}%   ◈ {t.tokens_estimate/1000:.1f}k/{t.context_limit/1000:.0f}k   "
            f"TOK {t.tokens_estimate:,}   TOOLS {t.tool_calls}   FAIL {t.tool_failures}   "
            f"TIME {t.elapsed:05.1f}s   MODE {provider}/{proto}"
        )
        mem = self.harness.memory.status if self.harness else "starting"
        workspace = str(self.settings.humoid_workspace.resolve())
        self.query_one("#session-info", Static).update(
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Messages: {t.messages}   Tools: {t.tool_calls}\n"
            f"Context: {t.context_percent}% ({t.tokens_estimate/1000:.1f}k/{t.context_limit/1000:.0f}k)\n"
            f"Workspace: {workspace}\nMemory: {mem}"
        )

    async def emit_event(self, event: AgentEvent):
        log = self.query_one("#activity-log", RichLog)
        chat = self.query_one("#chat", RichLog)
        stamp = time.strftime("%H:%M:%S")
        glyphs = {"tool_call":"↗", "tool_result":"✓", "error":"✗", "memory":"◆", "provider":"◈",
                  "protocol":"⌁", "profile":"◎", "round":"›", "reasoning":"◊", "gemma4":"G"}
        colors = {"tool_call":"cyan", "tool_result":"green", "error":"red", "reasoning":"magenta",
                  "provider":"blue", "memory":"green", "protocol":"cyan", "profile":"yellow"}
        if event.kind == "token":
            chat.write(event.message, expand=False)
            self.telemetry.tokens_estimate += max(1, len(event.message)//4)
        elif event.kind == "reasoning":
            if self.settings.humoid_show_reasoning:
                chat.write(f"[magenta]ASSISTANT (thinking)[/magenta]\n{escape(event.message)}")
        elif event.kind == "tool_call":
            self.telemetry.tool_calls += 1
            log.write(f"{stamp} [cyan]↗ TOOL.CALL[/cyan] {escape(event.detail.get('tool',''))}  id:{escape(event.detail.get('digest',''))}")
            chat.write(f"\n[bold red]TOOL.CALL[/bold red]\n[red]{escape(event.message)}[/red]\n")
        elif event.kind == "tool_result":
            if not event.detail.get("ok", False): self.telemetry.tool_failures += 1
            log.write(f"{stamp} [green]✓ TOOL.RESULT[/green] {escape(event.detail.get('tool',''))}")
            chat.write(f"[bold green]TOOL.RESULT[/bold green]\n[green]{escape(event.message)}[/green]\n")
        else:
            color = colors.get(event.kind, "white")
            glyph = glyphs.get(event.kind, "•")
            log.write(f"{stamp} [{color}]{glyph} {escape(event.kind.upper())}[/{color}] {escape(event.message)}")
        self._refresh_metrics()

    async def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip(); event.input.value = ""
        if not text or not self.harness: return
        chat = self.query_one("#chat", RichLog)
        self.telemetry.messages += 1
        self.telemetry.tokens_estimate += max(1, len(text)//4)
        chat.write(f"\n[bold blue]USER  {time.strftime('%H:%M:%S')}[/bold blue]\n{escape(text)}\n")
        if await self._command(text):
            self._refresh_metrics(); return
        if not self.harness_ready:
            chat.write("[yellow]Services are still initializing. F1 shows commands that remain available.[/yellow]")
            return
        if self.active_worker and not self.active_worker.is_finished:
            chat.write("[yellow]An operation is already active. Press Esc or Ctrl+C to cancel it.[/yellow]")
            return
        self.active_worker = self.run_worker(
            self._run_prompt(text), name="agent-request", exclusive=True,
        )

    async def _run_prompt(self, text: str) -> None:
        chat = self.query_one("#chat", RichLog)
        started = time.perf_counter()
        try:
            if (
                self.settings.humoid_gemma_autostart
                and self.settings.humoid_provider == "llamacpp"
            ):
                await self.ensure_local_gemma()
            response = await self.harness.run(text)
            if response.strip():
                chat.write("\n[green]✓ COMPLETE[/green]\n")
            else:
                chat.write("\n[yellow]NO RESPONSE — the model returned an empty turn[/yellow]\n")
        except asyncio.CancelledError:
            chat.write("\n[yellow]CANCELLED — provider/tool operation stopped[/yellow]\n")
            raise
        except Exception as exc:
            chat.write(f"\n[red]ERROR: {escape(str(exc))}[/red]\n")
        finally:
            self.telemetry.last_latency_ms = (time.perf_counter()-started)*1000
            self.query_one("#command", Input).focus(); self._refresh_metrics()

    async def _command(self, text: str) -> bool:
        chat = self.query_one("#chat", RichLog)
        if text in {"/help", "/?"}: self.action_help_reference(); return True
        if text == "/sessions": self.action_sessions(); return True
        if text == "/language": self.action_language(); return True
        if text == "/clear": self.action_clear_logs(); return True
        if text in {"/cancel"}: self.action_cancel_active(); return True
        if text == "/memory": self.action_memory(); return True
        if text.startswith("/memory search "):
            self.push_screen(MemoryBrowserScreen(text.removeprefix("/memory search ").strip())); return True
        if text == "/context": self.action_context(); return True
        if text == "/settings": self.action_model_settings(); return True
        if text.startswith("/shell "):
            mode = text.removeprefix("/shell ").strip().lower()
            if mode not in {"on", "off"}:
                chat.write("[red]Usage: /shell on|off[/red]")
            else:
                enabled = mode == "on"
                self.settings.humoid_allow_shell = enabled
                self.save_preference("HUMOID_ALLOW_SHELL", str(enabled).lower())
                chat.write(f"[yellow]Shell execution {'enabled' if enabled else 'disabled'}; cwd is the workspace, with OS-user permissions.[/yellow]")
            return True
        if text.startswith("/set "):
            parts = text.split(maxsplit=2)
            if len(parts) < 3: chat.write("[red]Usage: /set KEY VALUE[/red]")
            else:
                self.save_preference(parts[1], parts[2]); chat.write(f"[green]Saved {escape(parts[1].upper())}[/green]")
            return True
        if text == "/undo":
            chat.write(escape(await self.harness.tools.execute("undo_file_change", {}))); return True
        if text.startswith("/model "):
            await self._model_command(text.removeprefix("/model "), chat); return True
        if text.startswith("/provider "):
            name = text.split(maxsplit=1)[1]
            if name not in {"digitalocean","meta","openai","llamacpp","litert"}:
                chat.write("[red]Unknown provider[/red]"); return True
            await self.switch_active_provider(name); return True
        if text.startswith("/gemma4 "):
            self.harness.set_gemma4(mode=text.split(maxsplit=1)[1]); self._refresh_header(); return True
        if text.startswith("/gemma-thinking "):
            self.harness.set_gemma4(thinking=text.split(maxsplit=1)[1]); self._refresh_header(); return True
        if text.startswith("/autonomy "):
            mode=text.split(maxsplit=1)[1]
            if mode not in {"review","balanced","autonomous"}: chat.write("[red]review|balanced|autonomous[/red]")
            else: self.settings.humoid_autonomy_mode=mode; chat.write(f"[green]Autonomy: {mode}[/green]")
            return True
        if text == "/health":
            ok, status = await self.harness.provider.health(); chat.write(escape(status)); return True
        if text == "/profile":
            p=self.harness.profile; chat.write(f"[cyan]{p.key}: protocol={p.protocol} api={p.preferred_api} parallel={p.supports_parallel} programmatic={p.supports_programmatic_tools}[/cyan]"); return True
        return False

    async def _model_command(self, command: str, chat: RichLog) -> None:
        parts = command.split()
        action = parts[0] if parts else ""
        manager = self.model_manager
        try:
            if action == "detect":
                profile = await manager.detect_hardware(); chat.write(escape(str(profile)))
            elif action == "list":
                chat.write(escape("\n".join(path.name for path in manager.models()) or "No GGUF models downloaded"))
            elif action == "download" and len(parts) >= 3:
                chat.write(escape(str(await manager.download(parts[1], parts[2]))))
            elif action == "inspect" and len(parts) >= 2:
                chat.write(escape(str(await manager.inspect(manager.model_dir / Path(parts[1]).name))))
            elif action == "launch" and len(parts) >= 2:
                port = int(parts[2]) if len(parts) > 2 else 8080
                chat.write(escape(await manager.launch(manager.model_dir / Path(parts[1]).name, port)))
            elif action == "stop":
                chat.write(escape(await manager.stop()))
            elif action == "benchmark" and len(parts) >= 2:
                chat.write(escape(str(await manager.benchmark(manager.model_dir / Path(parts[1]).name))))
            else:
                chat.write("[red]Use: /model detect|list|download REPO FILE|inspect FILE|launch FILE [PORT]|stop|benchmark FILE[/red]")
        except Exception as exc:
            chat.write(f"[red]{escape(str(exc))}[/red]")

    def tr(self, key: str) -> str:
        return translate(self.language, key)

    def set_language(self, code: str) -> None:
        if code not in LOCALES:
            raise ValueError(f"Unsupported language: {code}")
        self.language = code
        self.settings.humoid_language = code
        self.save_preference("HUMOID_LANGUAGE", code)
        self._apply_language()

    def _apply_language(self) -> None:
        labels = {
            "nav-language": f"{self.language.upper()} · {self.tr('language')}",
            "nav-help": self.tr("help"), "nav-sessions": self.tr("sessions"),
            "nav-memory": self.tr("memory"), "nav-context": self.tr("context"),
            "nav-settings": self.tr("settings"),
        }
        for widget_id, label in labels.items():
            self.query_one(f"#{widget_id}", Button).label = label
        self.query_one("#activity-title", Static).update(self.tr("activity"))
        self.query_one("#agents-title", Static).update(self.tr("agents"))
        self.query_one("#session-title", Static).update(self.tr("session_info"))
        self._refresh_header()

    def save_current_session(self) -> None:
        if not self.harness or self.session not in self.sessions:
            return
        self.sessions[self.session]["messages"] = [dict(message) for message in self.harness.messages]
        self.sessions[self.session]["accordion"] = self.harness.context_accordion

    def _refresh_session_tabs(self) -> None:
        tabs = []
        for session_id, state in self.sessions.items():
            title = str(state["title"])
            tabs.append(f"[bold cyan]● {escape(title)}[/bold cyan]" if session_id == self.session else f"○ {escape(title)}")
        self.query_one("#session-tabs", Static).update("   ".join(tabs))

    def _render_session_messages(self) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        for message in self.harness.messages[1:]:
            role = str(message.get("role", "message")).upper()
            content = str(message.get("content") or "")
            if content:
                chat.write(f"[bold cyan]{escape(role)}[/bold cyan]\n{escape(content)}\n")

    def new_session(self, title: str | None = None) -> str:
        self.save_current_session()
        session_id = uuid.uuid4().hex[:8]
        system = dict(self.harness.messages[0])
        accordion = ContextAccordion(self.settings.context_limit())
        self.sessions[session_id] = {
            "title": title or f"Session {len(self.sessions) + 1}",
            "messages": [system], "accordion": accordion,
        }
        self.session = session_id
        self.harness.messages = [system]
        self.harness.context_accordion = accordion
        self.telemetry.messages = 0
        self._render_session_messages()
        self._refresh_header(); self._refresh_session_tabs()
        return session_id

    def switch_session(self, session_id: str) -> None:
        if session_id not in self.sessions:
            return
        self.save_current_session()
        state = self.sessions[session_id]
        self.session = session_id
        self.harness.messages = [dict(message) for message in state["messages"]]
        self.harness.context_accordion = state["accordion"]
        self.telemetry.messages = sum(1 for message in self.harness.messages if message.get("role") == "user")
        self._render_session_messages()
        self._refresh_header(); self._refresh_session_tabs(); self._refresh_metrics()

    def close_session(self, session_id: str) -> bool:
        if len(self.sessions) <= 1 or session_id not in self.sessions:
            return False
        was_active = session_id == self.session
        del self.sessions[session_id]
        if was_active:
            self.switch_session(next(iter(self.sessions)))
        self._refresh_session_tabs()
        return True

    def action_clear_logs(self):
        self.query_one("#activity-log", RichLog).clear(); self.query_one("#chat", RichLog).clear()
    def action_agents(self): self.query_one("#chat", RichLog).write("[cyan]Agent tree is visible at left. Multi-agent scheduler is staged for the next execution wave.[/cyan]")
    def action_gemma_menu(self): self.query_one("#chat", RichLog).write("[cyan]/gemma4 auto|native|off  /gemma-thinking off|low|on[/cyan]")
    def action_tools(self): self.query_one("#chat", RichLog).write("[cyan]Tools: list_files, read_file, write_file, run_command. Fenced by workspace policy.[/cyan]")
    def action_help_reference(self): self.push_screen(ReferenceScreen())
    def action_memory(self):
        if self.harness_ready: self.push_screen(MemoryBrowserScreen())
        else: self.notify("Memory is still initializing", severity="warning")
    def action_context(self): self.push_screen(ContextAccordionScreen())
    def action_settings(self): self.action_model_settings()
    def action_model_settings(self): self.push_screen(SettingsModelScreen())
    def action_language(self): self.push_screen(LanguageScreen())
    def action_sessions(self): self.push_screen(SessionsScreen())
    def action_cancel_active(self):
        if self.active_worker and not self.active_worker.is_finished:
            self.active_worker.cancel()
            self.notify("Cancelling active operation…", severity="warning")
        else:
            self.query_one("#command", Input).focus()

    def _navigation_buttons(self) -> list[Button]:
        return list(self.query("#bottom-nav Button"))

    def _navigation_focused(self) -> bool:
        return isinstance(self.focused, Button) and str(self.focused.id).startswith("nav-")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"navigation_left", "navigation_right", "leave_navigation"}:
            return self._navigation_focused()
        if action == "focus_navigation":
            return not self._navigation_focused() and not self.screen.is_modal
        return True

    def action_focus_navigation(self) -> None:
        self._navigation_buttons()[0].focus()

    def action_leave_navigation(self) -> None:
        self.query_one("#command", Input).focus()

    def _move_navigation(self, direction: int) -> None:
        buttons = self._navigation_buttons()
        if self.focused not in buttons:
            buttons[0].focus()
            return
        buttons[(buttons.index(self.focused) + direction) % len(buttons)].focus()

    def action_navigation_left(self) -> None:
        self._move_navigation(-1)

    def action_navigation_right(self) -> None:
        self._move_navigation(1)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = {
            "nav-language": self.action_language,
            "nav-help": self.action_help_reference,
            "nav-sessions": self.action_sessions,
            "nav-memory": self.action_memory,
            "nav-context": self.action_context,
            "nav-settings": self.action_model_settings,
        }.get(event.button.id or "")
        if action:
            action()

    async def on_unmount(self) -> None:
        await self.model_manager.stop()
        if self.harness:
            await self.harness.close()
