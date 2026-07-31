from .action_guard import install_action_guard

install_action_guard()

from .app import HumoidApp  # noqa: E402


def main():
    HumoidApp().run()


if __name__ == "__main__":
    main()
