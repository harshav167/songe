"""Bounded, resumable daytime reconstruction outside the serving process."""

import argparse
from pathlib import Path

from .config import Config
from .log import Store


def refresh(config, store):
    while True:
        changed, rows = store.reconstruct()
        if rows < 32 or not changed:
            break
    store.scoreboard_from_steps()
    # Rows recorded before repeated parts were shared are rewritten here, in
    # this guarded child, never in the serving process (recordings.dedupe).
    from .recordings import dedupe

    dedupe(store)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if not args.child:
        import sys

        from .memory import guarded_run

        raise SystemExit(
            guarded_run(
                [sys.executable, "-m", "songe.experience", "--child", *sys.argv[1:]]
            ).returncode
        )
    config = Config.load(args.data_dir)
    if args.model:
        config.public_model = args.model
    from .platform.locking import acquire

    store = Store(args.data_dir)
    with (store.root / "experience.lock").open("a+") as lease:
        acquire(lease)
        refresh(config, store)


if __name__ == "__main__":
    main()
