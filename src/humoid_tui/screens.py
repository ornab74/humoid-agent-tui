from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Input,
    Label,
    ProgressBar,
    Static,
    TextArea,
)

from .i18n import LOCALES

HELP_TEXT = """[bold cyan]KEYBOARD REFERENCE[/bold cyan]

From the command line, press Down to focus the bottom navigation bar.
Use Left/Right to highlight a page and Enter to open it. Press Up to return
to the command line. Every page has a top-left Back button.

Ctrl+1  Help                Ctrl+2  Memory browser
Ctrl+3  Context accordion   Ctrl+4  Settings + local models
F1/F5/F6/F12 remain secondary aliases when the terminal forwards them.
F2  Agents                  F3  Gemma 4       F4  Tools
Ctrl+K  Command palette     Ctrl+L  Clear logs
Esc / Ctrl+C  Cancel active model request or tool
Ctrl+Q  Quit                Tab / Shift+Tab  Move focus

[bold cyan]SLASH COMMANDS[/bold cyan]

/help                         Open this reference
/sessions                     Create, switch, rename, or close sessions
/language                     Change the TUI language
/clear                        Clear conversation and activity logs
/health                       Probe the active provider
/profile                      Show active model capability profile
/provider NAME                digitalocean|meta|openai|llamacpp|litert
/autonomy MODE                review|balanced|autonomous
/gemma4 MODE                  auto|native|off
/gemma-thinking LEVEL         off|low|on
/memory                       Open memory browser
/memory search QUERY          Search memory
/context                      Open expandable context history
/settings                     Open settings and model manager
/shell on|off                 Enable or disable workspace-fenced shell tools
/set KEY VALUE                Persist a value safely to .env
/model detect                 Detect CPU/GPU/CUDA
/model list                   List downloaded GGUF models
/model download REPO FILE     Download a GGUF from Hugging Face
/model inspect FILE           Inspect a local GGUF
/model launch FILE [PORT]     Start managed llama.cpp server
/model stop                   Stop managed server
/model benchmark FILE         Run a small local benchmark
/undo                         Undo the most recent agent file write
/cancel                       Cancel the active model/tool operation

File writes open a unified-diff preview. Approve, edit, or reject the proposed
content. Package install/uninstall buttons require an explicit click.
"""

HELP_CATEGORIES = {
    "all": HELP_TEXT,
    "navigation": "[bold cyan]NAVIGATION[/bold cyan]\n\nDown focuses the dashboard menu. Left/Right selects a page; Enter opens it.\nInside pages, Up/Down changes controls and Enter/Space activates them. Tab crosses editor and table boundaries.",
    "memory": "[bold cyan]MEMORY[/bold cyan]\n\nEnter/Space checks rows. Verify and Delete apply to checked rows. Save Edit applies to the highlighted row.\n/memory search QUERY opens filtered results.",
    "models": "[bold cyan]LOCAL MODELS[/bold cyan]\n\nDetect hardware before installing.\n/model list | download REPO FILE | inspect FILE | launch FILE [PORT] | stop | benchmark FILE",
    "tools": "[bold cyan]TOOLS AND SAFETY[/bold cyan]\n\nFile writes require diff approval. /undo restores the last write.\nEsc/Ctrl+C cancels active providers and subprocesses.",
}

NAV_HINT = "[dim]↑/↓ move  ←/→ move between buttons  Enter/Space activate  Tab enters editors/tables[/dim]"


class NavigableModal(ModalScreen):
    """Modal with predictable keyboard focus traversal."""

    BINDINGS: ClassVar = [
        Binding("down", "nav_next_vertical", "Next control", priority=True),
        Binding("up", "nav_previous_vertical", "Previous control", priority=True),
        Binding("right", "nav_next_horizontal", "Next control", priority=True),
        Binding("left", "nav_previous_horizontal", "Previous control", priority=True),
    ]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        focused = self.focused
        if action.startswith("nav_"):
            if isinstance(focused, DataTable) and action.endswith("horizontal"):
                return False
            if isinstance(focused, TextArea) and action.endswith("horizontal"):
                return False
            if isinstance(focused, Input) and action.endswith("horizontal"):
                return False
        return True

    def action_nav_next_vertical(self) -> None:
        if isinstance(self.focused, DataTable):
            table = self.focused
            if table.row_count and table.cursor_row < table.row_count - 1:
                table.action_cursor_down()
            else:
                self.focus_next()
            return
        self.focus_next()

    def action_nav_previous_vertical(self) -> None:
        if isinstance(self.focused, DataTable):
            table = self.focused
            if table.cursor_row > 0:
                table.action_cursor_up()
            else:
                self.focus_previous()
            return
        self.focus_previous()

    def action_nav_next_horizontal(self) -> None:
        self.focus_next()

    def action_nav_previous_horizontal(self) -> None:
        self.focus_previous()


class ReferenceScreen(NavigableModal):
    CSS = """
    ReferenceScreen { align: center middle; background: #0008; }
    #reference { width: 88%; height: 88%; border: round cyan; background: #050b12; padding: 1 2; }
    #close-reference { dock: bottom; width: 16; }
    """
    BINDINGS: ClassVar = [
        *NavigableModal.BINDINGS,
        Binding("escape", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="reference"):
            yield Button(self.app.tr("back"), id="back-reference")
            yield Static(NAV_HINT, markup=True)
            with Horizontal(classes="line"):
                yield Button("ALL", id="help-all", variant="primary")
                yield Button("NAVIGATION", id="help-navigation")
                yield Button("MEMORY", id="help-memory")
                yield Button("MODELS", id="help-models")
                yield Button("TOOLS", id="help-tools")
            yield Input(placeholder="Search reference, then press Enter…", id="help-search")
            yield Static(HELP_TEXT, markup=True, id="help-content")

    def on_mount(self) -> None:
        self.query_one("#back-reference", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-reference":
            self.dismiss()
            return
        category = str(event.button.id).removeprefix("help-")
        if category in HELP_CATEGORIES:
            self.query_one("#help-content", Static).update(HELP_CATEGORIES[category])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "help-search":
            return
        query = event.value.strip().lower()
        lines = [line for line in HELP_TEXT.splitlines() if not query or query in line.lower()]
        result = "\n".join(lines) if lines else "No matching commands."
        self.query_one("#help-content", Static).update(result)


class DiffApprovalScreen(ModalScreen[tuple[bool, str]]):
    CSS = """
    DiffApprovalScreen { align: center middle; background: #000a; }
    #diff-dialog { width: 92%; height: 90%; border: round yellow; background: #050b12; padding: 1; }
    #diff { height: 45%; border: solid #263744; }
    #edited-content { height: 1fr; border: solid #263744; }
    #diff-actions { height: 3; }
    """

    def __init__(self, path: str, diff: str, content: str) -> None:
        super().__init__()
        self.path, self.diff, self.content = path, diff, content

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="diff-dialog"):
            yield Label(f"FILE CHANGE PREVIEW: {self.path}")
            yield TextArea(self.diff, id="diff", read_only=True, language="diff")
            yield Label("Proposed content (editable before approval)")
            yield TextArea(self.content, id="edited-content")
            with Horizontal(id="diff-actions"):
                yield Button("Approve", id="approve", variant="success")
                yield Button("Approve edited", id="approve-edited", variant="primary")
                yield Button("Reject", id="reject", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve":
            self.dismiss((True, self.content))
        elif event.button.id == "approve-edited":
            self.dismiss((True, self.query_one("#edited-content", TextArea).text))
        else:
            self.dismiss((False, self.content))


class MemoryBrowserScreen(NavigableModal):
    BINDINGS: ClassVar = [
        *NavigableModal.BINDINGS,
        Binding("space", "toggle_memory", "Check memory", priority=True),
    ]
    CSS = """
    MemoryBrowserScreen { align: center middle; background: #000a; }
    #memory-dialog { width: 94%; height: 92%; border: round cyan; background: #050b12; padding: 1; }
    #memory-search { height: 3; }
    #memory-table { height: 42%; }
    #memory-editor { height: 1fr; border: solid #263744; }
    #memory-actions { height: 3; }
    """

    def __init__(self, initial_query: str = "") -> None:
        super().__init__()
        self.initial_query = initial_query
        self.selected_id = ""
        self.checked_ids: set[str] = set()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="memory-dialog"):
            yield Button(self.app.tr("back"), id="back-memory")
            yield Static(NAV_HINT, markup=True)
            yield Label("MEMORY BROWSER — Enter/Space checks a row; actions apply to checked rows")
            yield Input(value=self.initial_query, placeholder=f"{self.app.tr('search')}…", id="memory-search")
            yield DataTable(id="memory-table", cursor_type="row", zebra_stripes=True)
            yield TextArea(id="memory-editor")
            with Horizontal(id="memory-actions"):
                yield Button(self.app.tr("search"), id="search", variant="primary")
                yield Button("SAVE EDIT", id="save", variant="success")
                yield Button("VERIFY CHECKED", id="verify")
                yield Button("DELETE CHECKED", id="delete", variant="error")
                yield Button(self.app.tr("close"), id="close")

    def on_mount(self) -> None:
        table = self.query_one("#memory-table", DataTable)
        table.add_column("✓", key="checked", width=3)
        table.add_column("ID", key="id")
        table.add_column("Status", key="status")
        table.add_column("Tier", key="tier")
        table.add_column("Channel", key="channel")
        table.add_column("Created", key="created")
        table.add_column("Preview", key="preview")
        self.run_worker(self.refresh_memories(self.initial_query), exclusive=True)
        self.query_one("#back-memory", Button).focus()

    async def refresh_memories(self, query: str = "") -> None:
        memory = self.app.harness.memory
        hits = await (memory.search(query, limit=200) if query else memory.list_memories(200))
        table = self.query_one("#memory-table", DataTable)
        table.clear()
        for hit in hits:
            table.add_row(
                "✓" if hit.memory_id in self.checked_ids else " ",
                hit.memory_id[:10], hit.validation_status, hit.memory_tier, hit.channel,
                hit.created_at[:19], " ".join(hit.text.split())[:100], key=hit.memory_id,
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "memory-search":
            await self.refresh_memories(event.value.strip())
            self.query_one("#memory-table", DataTable).focus()

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        await self._toggle_memory(str(event.row_key.value))

    async def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.selected_id = str(event.row_key.value)
        hits = await self.app.harness.memory.list_memories(500)
        hit = next((item for item in hits if item.memory_id == self.selected_id), None)
        if hit:
            self.query_one("#memory-editor", TextArea).load_text(hit.text)

    async def action_toggle_memory(self) -> None:
        table = self.query_one("#memory-table", DataTable)
        if self.focused is not table or not table.row_count:
            return
        row_key, _column_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        await self._toggle_memory(str(row_key.value))

    async def _toggle_memory(self, memory_id: str) -> None:
        if memory_id in self.checked_ids:
            self.checked_ids.remove(memory_id)
            marker = " "
        else:
            self.checked_ids.add(memory_id)
            marker = "✓"
        self.selected_id = memory_id
        self.query_one("#memory-table", DataTable).update_cell(memory_id, "checked", marker)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        query = self.query_one("#memory-search", Input).value.strip()
        if button in {"close", "back-memory"}:
            self.dismiss()
        elif button == "search":
            await self.refresh_memories(query)
        elif button in {"verify", "delete"} and not self.checked_ids:
            self.notify("Check one or more memories with Enter or Space first", severity="warning")
        elif button == "save" and not self.selected_id:
            self.notify("Highlight a memory first", severity="warning")
        elif button == "save":
            await self.app.harness.memory.update_memory(
                self.selected_id, text=self.query_one("#memory-editor", TextArea).text,
            )
            await self.refresh_memories(query)
            self.notify("Memory updated")
        elif button == "verify":
            for memory_id in tuple(self.checked_ids):
                await self.app.harness.memory.update_memory(memory_id, validation_status="verified")
            await self.refresh_memories(query)
            self.notify(f"Verified {len(self.checked_ids)} checked memories")
        elif button == "delete":
            deleted = len(self.checked_ids)
            for memory_id in tuple(self.checked_ids):
                await self.app.harness.memory.delete_memory(memory_id)
            self.checked_ids.clear()
            self.selected_id = ""
            self.query_one("#memory-editor", TextArea).load_text("")
            await self.refresh_memories(query)
            self.notify(f"Deleted {deleted} checked memories")


class ContextAccordionScreen(NavigableModal):
    CSS = """
    ContextAccordionScreen { align: center middle; background: #000a; }
    #context-dialog { width: 92%; height: 90%; border: round magenta; background: #050b12; padding: 1; }
    """

    def compose(self) -> ComposeResult:
        accordion = self.app.harness.context_accordion
        with VerticalScroll(id="context-dialog"):
            yield Button(self.app.tr("back"), id="back-context")
            yield Static(NAV_HINT, markup=True)
            yield Label("CONTEXT ACCORDION — folded summaries retain expandable source")
            yield Static("", id="context-status")
            with Horizontal(classes="line"):
                yield Button("REFRESH STATUS", id="context-refresh", variant="primary")
                yield Button("EXPAND ALL", id="context-expand")
                yield Button("COLLAPSE ALL", id="context-collapse")
            yield Collapsible(
                Static(f"Messages: {len(self.app.harness.messages)}\nEstimated tokens: {accordion.estimate(self.app.harness.messages)}"),
                title="Active model context", collapsed=False,
            )
            for index, archive in enumerate(accordion.archives):
                yield Collapsible(
                    Static(accordion.render_archive(archive, index)),
                    title=f"Folded archive #{index} ({len(archive.original_messages)} source messages)",
                )
                yield Collapsible(
                    Static("\n\n".join(f"{m.get('role')}: {m.get('content')}" for m in archive.original_messages)),
                    title=f"Expanded source #{index}",
                )
            yield Button(self.app.tr("close"), id="close-context")

    def on_mount(self) -> None:
        self.query_one("#back-context", Button).focus()
        self._refresh_status()

    def _refresh_status(self) -> None:
        accordion = self.app.harness.context_accordion
        tokens = accordion.estimate(self.app.harness.messages)
        percent = min(100, int(tokens / max(1, accordion.context_limit) * 100))
        self.query_one("#context-status", Static).update(
            f"Active: {tokens:,}/{accordion.context_limit:,} estimated tokens ({percent}%)  "
            f"Folded archives: {len(accordion.archives)}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        if action in {"back-context", "close-context"}:
            self.dismiss()
        elif action == "context-refresh":
            self._refresh_status()
        elif action == "context-expand":
            for section in self.query(Collapsible):
                section.collapsed = False
            self.notify("Expanded all context sections")
        elif action == "context-collapse":
            for section in self.query(Collapsible):
                section.collapsed = True
            self.notify("Collapsed all context sections")


class LanguageScreen(NavigableModal):
    CSS = """
    LanguageScreen { align: center middle; background: #000a; }
    #language-dialog { width: 78%; height: 88%; border: round green; background: #050b12; padding: 1; }
    #language-table { height: 1fr; border: solid #263744; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="language-dialog"):
            yield Button(self.app.tr("back"), id="back-language")
            yield Static(NAV_HINT, markup=True)
            yield Label(self.app.tr("select_language"), id="language-title")
            yield DataTable(id="language-table", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#language-table", DataTable)
        table.add_column("✓", key="selected", width=3)
        table.add_column("Language", key="language")
        table.add_column("Code", key="code", width=8)
        table.add_column("Region", key="region")
        for locale in LOCALES.values():
            table.add_row(
                "✓" if locale.code == self.app.language else "",
                locale.native_name, locale.code, locale.region, key=locale.code,
            )
        self.query_one("#back-language", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-language":
            self.dismiss()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        code = str(event.row_key.value)
        self.app.set_language(code)
        table = self.query_one("#language-table", DataTable)
        for row_key in table.rows:
            table.update_cell(row_key, "selected", "✓" if str(row_key.value) == code else "")
        self.query_one("#language-title", Label).update(self.app.tr("select_language"))
        self.query_one("#back-language", Button).label = self.app.tr("back")
        self.notify(f"{LOCALES[code].native_name} applied and saved")


class SessionsScreen(NavigableModal):
    CSS = """
    SessionsScreen { align: center middle; background: #000a; }
    #sessions-dialog { width: 82%; height: 86%; border: round cyan; background: #050b12; padding: 1; }
    #sessions-table { height: 1fr; border: solid #263744; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="sessions-dialog"):
            yield Button(self.app.tr("back"), id="back-sessions")
            yield Static(NAV_HINT, markup=True)
            yield Label(self.app.tr("sessions"))
            yield DataTable(id="sessions-table", cursor_type="row", zebra_stripes=True)
            yield Input(placeholder="Session name", id="session-name")
            with Horizontal(classes="line"):
                yield Button(self.app.tr("new"), id="session-new", variant="success")
                yield Button(self.app.tr("switch"), id="session-switch", variant="primary")
                yield Button(self.app.tr("rename"), id="session-rename")
                yield Button(self.app.tr("close"), id="session-close", variant="error")

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_column("Active", key="active", width=7)
        table.add_column("Name", key="name")
        table.add_column("Messages", key="messages", width=10)
        table.add_column("ID", key="id")
        self.refresh_sessions()
        self.query_one("#back-sessions", Button).focus()

    def refresh_sessions(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.clear()
        self.app.save_current_session()
        for session_id, state in self.app.sessions.items():
            table.add_row(
                "✓" if session_id == self.app.session else "",
                state["title"], str(len(state["messages"])), session_id, key=session_id,
            )

    def _selected(self) -> str:
        table = self.query_one("#sessions-table", DataTable)
        if not table.row_count:
            return ""
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        return str(row_key.value)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        state = self.app.sessions.get(str(event.row_key.value))
        if state:
            self.query_one("#session-name", Input).value = state["title"]

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.switch_session(str(event.row_key.value))
        self.refresh_sessions()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        name = self.query_one("#session-name", Input).value.strip()
        selected = self._selected()
        if action == "back-sessions":
            self.dismiss()
        elif action == "session-new":
            self.app.new_session(name or None)
            self.refresh_sessions()
        elif action == "session-switch" and selected:
            self.app.switch_session(selected)
            self.refresh_sessions()
        elif action == "session-rename" and selected and name:
            self.app.sessions[selected]["title"] = name
            self.app._refresh_session_tabs()
            self.refresh_sessions()
        elif action == "session-close" and selected:
            if not self.app.close_session(selected):
                self.notify("At least one session must remain", severity="warning")
            self.refresh_sessions()


class SettingsModelScreen(NavigableModal):
    CSS = """
    SettingsModelScreen { align: center middle; background: #000a; }
    #settings-dialog { width: 92%; height: 94%; border: round cyan; background: #050b12; padding: 1 2; }
    #settings-title { color: #10d9ed; text-style: bold; margin-top: 1; }
    #settings-intro, .section-help { color: #91a4b7; margin-bottom: 1; }
    .settings-section { border: solid #263744; margin-top: 1; padding: 0 1; }
    .line { height: 3; }
    #active-provider { min-height: 3; background: #0a1822; color: #d9f7ff; padding: 0 1; }
    #shell-status, #gemma-autostart-status { min-height: 2; color: #b8c7d4; }
    #hardware { min-height: 7; background: #07111a; padding: 0 1; }
    #model-table { height: 9; border: solid #263744; }
    #download-progress { height: 1; }
    #download-detail { min-height: 2; color: #10d9ed; }
    #model-output { min-height: 6; border: solid #263744; background: #02070b; padding: 0 1; }
    #close-settings { margin-top: 1; width: 20; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="settings-dialog"):
            yield Button(self.app.tr("back"), id="back-settings")
            yield Static(NAV_HINT, markup=True)
            yield Label("SETTINGS & MODEL CONTROL", id="settings-title")
            yield Static(
                "Choose the active AI provider, manage local Gemma, and tune runtime behavior. "
                "Changes marked as persistent are saved to SQLite preferences and .env.",
                id="settings-intro",
            )

            yield Label("1  ACTIVE MODEL")
            yield Static(
                "Switch between remote GLM 5.2 and local Gemma. Starting Gemma can take a minute on CPU.",
                classes="section-help",
            )
            yield Static("", id="active-provider")
            with Horizontal(classes="line"):
                yield Button("USE GLM 5.2 (REMOTE)", id="use-remote", variant="primary")
                yield Button("START + USE GEMMA", id="start-gemma", variant="success")

            with Collapsible(title="2  Local Gemma server", collapsed=False, classes="settings-section"):
                yield Static(
                    "Control the llama.cpp server and decide whether it should start automatically when Gemma is selected.",
                    classes="section-help",
                )
                yield Static("", id="gemma-autostart-status")
                with Horizontal(classes="line"):
                    yield Button("USE RUNNING GEMMA", id="use-local", variant="success")
                    yield Button("START SERVER ONLY", id="launch")
                    yield Button("STOP SERVER", id="stop", variant="error")
                with Horizontal(classes="line"):
                    yield Button("ENABLE AUTO-START", id="gemma-autostart-on", variant="success")
                    yield Button("DISABLE AUTO-START", id="gemma-autostart-off", variant="warning")

            with Collapsible(title="3  Downloaded GGUF models", collapsed=False, classes="settings-section"):
                yield Static(
                    "Select a downloaded model, or provide a Hugging Face repository and filename to download one.",
                    classes="section-help",
                )
                yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
                yield Label("Hugging Face repository")
                yield Input(value="google/gemma-4-E2B-it-qat-q4_0-gguf", id="model-repo")
                yield Label("GGUF filename / selected local model")
                yield Input(value="gemma-4-E2B_q4_0-it.gguf", id="model-file")
                yield ProgressBar(total=100, show_eta=True, id="download-progress")
                yield Static("Download idle", id="download-detail")
                with Horizontal(classes="line"):
                    yield Button("REFRESH MODELS", id="refresh-models")
                    yield Button("DOWNLOAD", id="download", variant="primary")
                    yield Button("CANCEL DOWNLOAD", id="cancel-download", variant="error", disabled=True)
                    yield Button("INSPECT", id="inspect")
                    yield Button("BENCHMARK", id="benchmark")

            with Collapsible(title="4  Hardware & local runtime", collapsed=True, classes="settings-section"):
                yield Static(
                    "Detect acceleration support and manage the optional llama-cpp-python server package.",
                    classes="section-help",
                )
                yield Static("Detecting hardware…", id="hardware")
                with Horizontal(classes="line"):
                    yield Button("DETECT AGAIN", id="detect")
                    yield Button("INSTALL RUNTIME", id="install", variant="success")
                    yield Button("REINSTALL", id="reinstall", variant="warning")
                    yield Button("UNINSTALL", id="uninstall", variant="error")

            with Collapsible(title="5  Context windows", collapsed=True, classes="settings-section"):
                yield Static(
                    "Larger windows remember more conversation but consume more RAM and run more slowly.",
                    classes="section-help",
                )
                yield Label("Local Gemma context tokens")
                yield Input(
                    value=str(self.app.settings.humoid_gemma_context_limit),
                    placeholder="Gemma context", id="gemma-context", type="integer",
                )
                yield Label("Remote GLM 5.2 context tokens")
                yield Input(
                    value=str(self.app.settings.humoid_glm_context_limit),
                    placeholder="GLM 5.2 context", id="glm-context", type="integer",
                )
                yield Button("SAVE CONTEXT WINDOWS", id="apply-context", variant="primary")

            with Collapsible(title="6  Safety & advanced preferences", collapsed=True, classes="settings-section"):
                yield Static("", id="shell-status")
                with Horizontal(classes="line"):
                    yield Button("ENABLE SHELL TOOLS", id="shell-enable", variant="warning")
                    yield Button("DISABLE SHELL TOOLS", id="shell-disable", variant="error")
                yield Static(
                    "Advanced: persist any recognized environment setting. Most users do not need this.",
                    classes="section-help",
                )
                yield Input(placeholder="SETTING KEY (for example HUMOID_TEMPERATURE)", id="setting-key")
                yield Input(placeholder="SETTING VALUE", id="setting-value")
                yield Button("SAVE ADVANCED SETTING", id="save-setting", variant="primary")

            yield Label("ACTIVITY & RESULTS")
            yield Static("", id="model-output")
            yield Button(self.app.tr("close"), id="close-settings")

    def on_mount(self) -> None:
        self.hardware = None
        self.download_worker = None
        self.benchmark_worker = None
        self.provider_worker = None
        table = self.query_one("#model-table", DataTable)
        table.add_column("Selected", key="selected", width=8)
        table.add_column("GGUF model", key="name")
        table.add_column("Size", key="size")
        self.refresh_models()
        self.refresh_shell_status()
        self.refresh_gemma_autostart()
        self.refresh_active_provider()
        self.run_worker(self.detect(), exclusive=True)
        self.query_one("#back-settings", Button).focus()

    def refresh_models(self) -> None:
        table = self.query_one("#model-table", DataTable)
        table.clear()
        selected = Path(self.query_one("#model-file", Input).value).name
        for path in self.app.model_manager.models():
            size = f"{path.stat().st_size / (1024 ** 3):.2f} GiB"
            table.add_row("✓" if path.name == selected else "", path.name, size, key=path.name)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "model-table":
            return
        filename = str(event.row_key.value)
        self.query_one("#model-file", Input).value = filename

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "model-table":
            return
        filename = str(event.row_key.value)
        self.query_one("#model-file", Input).value = filename
        table = self.query_one("#model-table", DataTable)
        for row_key in table.rows:
            table.update_cell(row_key, "selected", "✓" if str(row_key.value) == filename else "")
        self.notify(f"Selected {filename}")

    async def detect(self) -> None:
        self.hardware = await self.app.model_manager.detect_hardware()
        manager = self.app.model_manager
        wheel = " ".join(manager.install_command(self.hardware))
        self.query_one("#hardware", Static).update(
            f"Backend: {self.hardware.backend}\nDevice: {self.hardware.name}\n"
            f"CUDA: {self.hardware.cuda_version or 'n/a'}\n"
            f"llama-cpp-python: {manager.package_version()}\nRecommended: {wheel}"
            f"\nLocal server:\n{manager.status_json()}"
        )

    def output(self, text: object) -> None:
        self.query_one("#model-output", Static).update(str(text))

    def refresh_active_provider(self) -> None:
        cfg = self.app.harness.provider.cfg if self.app.harness else self.app.settings.provider()
        self.query_one("#active-provider", Static).update(
            f"Active provider: {cfg.name}\nModel: {cfg.model}\nEndpoint: {cfg.base_url}"
        )

    def refresh_gemma_autostart(self) -> None:
        enabled = self.app.settings.humoid_gemma_autostart
        self.query_one("#gemma-autostart-status", Static).update(
            "Auto-start is ENABLED: Gemma starts when the saved provider is llama.cpp."
            if enabled else
            "Auto-start is DISABLED: start Gemma manually when you need it."
        )
        self.query_one("#gemma-autostart-on", Button).disabled = enabled
        self.query_one("#gemma-autostart-off", Button).disabled = not enabled

    async def activate_provider(self, name: str, path: Path) -> None:
        remote = self.query_one("#use-remote", Button)
        local = self.query_one("#use-local", Button)
        remote.disabled = local.disabled = True
        try:
            if name == "llamacpp":
                if not path.is_file():
                    raise RuntimeError("Select or download a GGUF model first")
                manager = self.app.model_manager
                self.output(f"Starting {path.name}… This may take a minute on CPU.")
                if not manager.process or manager.process.returncode is not None:
                    await manager.launch(
                        path, context=self.app.settings.humoid_gemma_context_limit,
                    )
                await manager.wait_until_ready()
                self.app.settings.llamacpp_model = path.stem
                self.app.settings.llamacpp_base_url = "http://127.0.0.1:8080/v1"
                self.app.save_preference("LLAMACPP_MODEL", path.stem)
                self.app.save_preference("LLAMACPP_BASE_URL", self.app.settings.llamacpp_base_url)
            await self.app.switch_active_provider(name)
            self.refresh_active_provider()
            self.output(f"Now using {self.app.harness.provider.cfg.model} ({name})")
        except Exception as exc:  # noqa: BLE001 - show provider/server failures in the modal
            self.output(f"Could not switch model: {exc}")
        finally:
            remote.disabled = local.disabled = False

    @staticmethod
    def _bytes(value: float) -> str:
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        index = 0
        while value >= 1024 and index < len(units) - 1:
            value /= 1024
            index += 1
        return f"{value:.1f} {units[index]}"

    def update_download_progress(self, status: dict[str, float | int | str]) -> None:
        downloaded = int(status.get("downloaded", 0))
        total = int(status.get("total", 0))
        rate = float(status.get("rate", 0.0))
        phase = str(status.get("phase", "Downloading"))
        percent = (downloaded / total * 100.0) if total else 0.0
        self.query_one("#download-progress", ProgressBar).update(total=100, progress=percent)
        total_text = self._bytes(total) if total else "unknown"
        self.query_one("#download-detail", Static).update(
            f"{phase}: {status.get('filename', '')}\n"
            f"{self._bytes(downloaded)} / {total_text}  {percent:5.1f}%  {self._bytes(rate)}/s"
        )

    async def download_model(self, repo: str, filename: str) -> None:
        self.query_one("#download", Button).disabled = True
        self.query_one("#cancel-download", Button).disabled = False
        self.query_one("#download-progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#download-detail", Static).update(f"Preparing {filename}…")
        try:
            path = await self.app.model_manager.download(
                repo, filename, progress=self.update_download_progress,
            )
            self.query_one("#download-progress", ProgressBar).update(total=100, progress=100)
            self.query_one("#download-detail", Static).update(f"Complete: {path}")
            self.output(path)
            self.refresh_models()
        except Exception as exc:
            self.query_one("#download-detail", Static).update(f"Download stopped: {exc}")
            self.output(f"{type(exc).__name__}: {exc}")
        finally:
            self.query_one("#download", Button).disabled = False
            self.query_one("#cancel-download", Button).disabled = True

    async def benchmark_model(self, path: Path) -> None:
        button = self.query_one("#benchmark", Button)
        button.disabled = True
        self.output(f"Benchmarking {path.name}… Loading the model may take a minute on CPU.")
        try:
            self.output(await self.app.model_manager.benchmark(path))
        finally:
            button.disabled = False

    def refresh_shell_status(self) -> None:
        enabled = self.app.settings.humoid_allow_shell
        self.query_one("#shell-status", Static).update(
            f"Shell tools: {'ENABLED' if enabled else 'DISABLED'}  "
            f"timeout={self.app.settings.humoid_shell_timeout_seconds}s  cwd=HUMOID_WORKSPACE"
        )
        self.query_one("#shell-enable", Button).disabled = enabled
        self.query_one("#shell-disable", Button).disabled = not enabled

    def apply_context_windows(self) -> None:
        gemma = int(self.query_one("#gemma-context", Input).value)
        glm = int(self.query_one("#glm-context", Input).value)
        if gemma < 1024 or glm < 1024:
            raise ValueError("Context windows must be at least 1,024 tokens")
        self.app.settings.humoid_gemma_context_limit = gemma
        self.app.settings.humoid_glm_context_limit = glm
        self.app.save_preference("HUMOID_GEMMA_CONTEXT_LIMIT", str(gemma))
        self.app.save_preference("HUMOID_GLM_CONTEXT_LIMIT", str(glm))
        active = self.app.settings.context_limit()
        self.app.telemetry.context_limit = active
        self.app.harness.context_accordion.context_limit = active
        for state in self.app.sessions.values():
            state["accordion"].context_limit = active
        self.output(
            f"Context windows saved: Gemma {gemma:,}; GLM 5.2 {glm:,}. "
            "Restart a running local Gemma server to change its allocated context."
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        manager = self.app.model_manager
        filename = self.query_one("#model-file", Input).value.strip()
        path = manager.model_dir / Path(filename).name
        if action in {"close-settings", "back-settings"}:
            self.app.model_manager.cancel_download()
            self.dismiss()
        elif action == "cancel-download":
            manager.cancel_download()
            self.query_one("#download-detail", Static).update("Cancelling download…")
        elif action in {"use-remote", "use-local"}:
            provider = "digitalocean" if action == "use-remote" else "llamacpp"
            self.provider_worker = self.run_worker(
                self.activate_provider(provider, path),
                name="provider-switch",
                exclusive=True,
            )
        elif action == "start-gemma":
            self.app.settings.llamacpp_model = path.stem
            self.app.save_preference("LLAMACPP_MODEL", path.stem)
            self.app.settings.humoid_gemma_autostart = True
            self.app.save_preference("HUMOID_GEMMA_AUTOSTART", "true")
            self.refresh_gemma_autostart()
            self.provider_worker = self.run_worker(
                self.start_and_activate_gemma(),
                name="start-gemma",
                exclusive=True,
            )
        elif action in {"gemma-autostart-on", "gemma-autostart-off"}:
            enabled = action == "gemma-autostart-on"
            self.app.settings.humoid_gemma_autostart = enabled
            self.app.save_preference("HUMOID_GEMMA_AUTOSTART", str(enabled).lower())
            self.refresh_gemma_autostart()
            self.output(
                f"Local Gemma auto-start {'enabled' if enabled else 'disabled'} "
                "and saved to .env"
            )
        elif action in {"shell-enable", "shell-disable"}:
            enabled = action == "shell-enable"
            self.app.settings.humoid_allow_shell = enabled
            self.app.save_preference("HUMOID_ALLOW_SHELL", str(enabled).lower())
            self.refresh_shell_status()
            self.output(
                f"Shell execution {'enabled' if enabled else 'disabled'} immediately "
                "and saved to .env. Commands start in HUMOID_WORKSPACE but retain OS-user permissions."
            )
        elif action == "apply-context":
            try:
                self.apply_context_windows()
            except ValueError as exc:
                self.output(str(exc))
        elif action == "refresh-models":
            self.refresh_models()
            self.output(f"Found {len(manager.models())} local GGUF models")
        elif action == "detect":
            await self.detect()
        elif action in {"install", "reinstall", "uninstall"}:
            if self.hardware is None:
                await self.detect()
            self.output(f"Running {action}; this can take several minutes…")
            code, output = await manager.mutate_package(action, self.hardware)
            self.output(f"exit={code}\n{output[-5000:]}")
            await self.detect()
        elif action == "save-setting":
            key = self.query_one("#setting-key", Input).value
            value = self.query_one("#setting-value", Input).value
            self.app.save_preference(key, value)
            self.output(f"Saved {key.upper()} to preferences.sqlite3 and .env.")
        elif action == "download":
            repo = self.query_one("#model-repo", Input).value.strip()
            self.download_worker = self.run_worker(
                self.download_model(repo, filename),
                name="model-download",
                exclusive=False,
            )
        elif action == "inspect":
            self.output(await manager.inspect(path))
        elif action == "launch":
            try:
                result = await manager.launch(
                    path, context=self.app.settings.humoid_gemma_context_limit,
                )
                self.output(result)
                self.app.save_preference("LLAMACPP_MODEL", path.stem)
                self.app.save_preference("LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1")
                await self.detect()
            except Exception as exc:  # noqa: BLE001 - keep model failures inside the modal
                self.output(f"Could not launch local server: {type(exc).__name__}: {exc}")
        elif action == "stop":
            self.output(await manager.stop())
        elif action == "benchmark":
            self.benchmark_worker = self.run_worker(
                self.benchmark_model(path),
                name="model-benchmark",
                exclusive=True,
            )

    async def start_and_activate_gemma(self) -> None:
        try:
            self.output("Starting server and activating local Gemma…")
            self.output(await self.app.ensure_local_gemma())
            self.refresh_active_provider()
            await self.detect()
        except Exception as exc:  # noqa: BLE001 - surface startup errors in the modal
            self.output(f"Could not start local Gemma: {type(exc).__name__}: {exc}")
