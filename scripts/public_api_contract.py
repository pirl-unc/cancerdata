#!/usr/bin/env python3
"""Check or deliberately update oncoref's public API compatibility manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oncoref
from oncoref.api_contract import (
    current_public_api_manifest,
    load_public_api_contract,
    public_api_compatibility_errors,
    public_api_contract_path,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the current additive API")
    parser.add_argument(
        "--allow-breaking",
        metavar="REVIEW_REFERENCE",
        help="approve a deliberate incompatible update and record its issue/PR reference",
    )
    args = parser.parse_args(argv)
    if args.allow_breaking and not args.update:
        parser.error("--allow-breaking requires --update")

    expected = load_public_api_contract()
    current = current_public_api_manifest()
    errors = public_api_compatibility_errors(expected, current)
    if not args.update:
        if errors:
            print("incompatible public API changes:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("public API contract compatible")
        return 0

    if errors and not args.allow_breaking:
        print("refusing to overwrite incompatible public API changes", file=sys.stderr)
        print("review them, then pass --allow-breaking ISSUE_OR_PR if intentional", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    current["generated_for_package"] = oncoref.__version__
    current["breaking_change_approval"] = args.allow_breaking
    path = public_api_contract_path()
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
