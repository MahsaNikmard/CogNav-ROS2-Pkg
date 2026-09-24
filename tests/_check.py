"""Minimal pass/fail reporter shared by the tests."""


class Checker:
    def __init__(self):
        self.ok = True

    def __call__(self, label, cond, detail=""):
        self.ok = self.ok and bool(cond)
        print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    def finish(self, name):
        print("\n" + (f"ALL PASS: {name}" if self.ok else "FAILURES PRESENT"))
        raise SystemExit(0 if self.ok else 1)
