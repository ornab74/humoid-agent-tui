from .edit_cycle_runtime import install_edit_cycle_runtime

install_edit_cycle_runtime()

from .app import HumoidApp  # noqa: E402  (runtime policy installs before app imports)


def main():
    HumoidApp().run()


if __name__ == "__main__":
    main()
