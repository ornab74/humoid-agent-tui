from .edit_cycle_runtime import install_edit_cycle_runtime
from .code_intelligence_runtime import install_code_intelligence_runtime
from .action_guard import install_action_guard

# Runtime order matters. The edit-cycle policy establishes evidence gates,
# impact intelligence enriches those cycles, and the action guard remains the
# outer bounded recovery layer for provider/tool protocol stalls.
install_edit_cycle_runtime()
install_code_intelligence_runtime()
install_action_guard()

from .app import HumoidApp  # noqa: E402


def main():
    HumoidApp().run()


if __name__ == "__main__":
    main()
