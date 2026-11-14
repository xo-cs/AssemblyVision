"""Users of the interface login (Week 11; ui/auth.py). The file is .config/ui_users.json
(never in git); passwords are stored only as scrypt hashes.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.ui_users add NAME        # asks for the password twice
    python -m assemblyvision.tools.ui_users list
    python -m assemblyvision.tools.ui_users remove NAME
"""
from __future__ import annotations

import argparse
import getpass
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.ui_users")
    p.add_argument("action", choices=("add", "list", "remove"))
    p.add_argument("name", nargs="?")
    args = p.parse_args(argv)
    from ..ui.auth import Users
    users = Users()
    if args.action == "list":
        print("\n".join(users.names()) or "no users")
        return 0
    if not args.name:
        p.error(f"{args.action} needs NAME")
    if args.action == "remove":
        print("removed" if users.remove(args.name) else "no such user")
        return 0
    pw = getpass.getpass(f"password for {args.name}: ")
    if pw != getpass.getpass("again: "):
        print("the passwords differ", file=sys.stderr)
        return 1
    try:
        users.add(args.name, pw)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"user {args.name} saved in {users.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
