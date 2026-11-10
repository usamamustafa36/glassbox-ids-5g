"""Presentation layer for the live demonstration.

The user watches this output, so every stage announces what it is about to
do, reports progress while it works, and prints how long it actually took.
Falls back to plain ASCII printing if ``rich`` is unavailable or the terminal
cannot render Unicode (older Windows consoles).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

try:                                        # pragma: no cover - import guard
    from rich.console import Console as _RichConsole
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
except Exception:                           # pragma: no cover
    _HAS_RICH = False


# Older Windows terminals default to cp1252 and raise on box-drawing glyphs.
def _unicode_ok() -> bool:
    if os.environ.get("XIDS_ASCII"):
        return False
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


_console = _RichConsole(highlight=False, soft_wrap=False) if _HAS_RICH else None

# Timings collected during the run, written out by the summary step.
_TIMINGS: list[dict[str, Any]] = []


def _emit(msg: str = "", style: str | None = None) -> None:
    if _console is not None:
        _console.print(msg, style=style)
    else:                                   # pragma: no cover
        print(msg)


# --------------------------------------------------------------------------
# Banners and section headers
# --------------------------------------------------------------------------
def banner(title: str, subtitle: str = "") -> None:
    """Big header printed once when a pipeline stage starts."""
    if _console is not None and _unicode_ok():
        body = Text(title, style="bold white")
        if subtitle:
            body.append("\n" + subtitle, style="dim")
        _console.print(Panel(body, border_style="cyan", padding=(0, 2)))
    else:                                   # pragma: no cover
        line = "=" * 72
        print(f"\n{line}\n{title}\n" + (f"{subtitle}\n" if subtitle else "") + line)


def rule(text: str = "") -> None:
    if _console is not None and _unicode_ok():
        _console.rule(f"[bold cyan]{text}" if text else "")
    else:                                   # pragma: no cover
        print("-" * 72 + (f" {text}" if text else ""))


def step(msg: str) -> None:
    """Announce the operation that is starting right now."""
    _emit(f"  [cyan]>[/cyan] {msg}" if _HAS_RICH else f"  > {msg}")


def info(msg: str) -> None:
    _emit(f"    [dim]{msg}[/dim]" if _HAS_RICH else f"    {msg}")


def ok(msg: str) -> None:
    mark = "[green]OK[/green]" if _HAS_RICH else "OK"
    _emit(f"  [{mark}] {msg}" if _HAS_RICH else f"  [OK] {msg}")


def warn(msg: str) -> None:
    _emit(f"  [yellow]![/yellow] {msg}" if _HAS_RICH else f"  [!] {msg}")


def fail(msg: str) -> None:
    _emit(f"  [red]x[/red] {msg}" if _HAS_RICH else f"  [x] {msg}")


def note(msg: str) -> None:
    """A finding worth the audience's attention."""
    if _console is not None:
        _console.print(f"  [bold yellow]FINDING[/bold yellow]  {msg}")
    else:                                   # pragma: no cover
        print(f"  FINDING  {msg}")


def blank() -> None:
    _emit("")


def saved(path: Path) -> None:
    """Tell the audience exactly where an artifact landed."""
    try:
        rel = path.relative_to(Path(__file__).resolve().parents[2])
    except ValueError:
        rel = path
    _emit(f"    [dim]saved ->[/dim] [blue]{rel}[/blue]" if _HAS_RICH
          else f"    saved -> {rel}")


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def _fmt_secs(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(s, 60)
    return f"{int(m)} min {sec:.0f} s"


@contextlib.contextmanager
def timed(label: str, estimate: str | None = None) -> Iterator[dict[str, Any]]:
    """Time a block, announcing start and finish.

    Yields a dict the caller can drop extra facts into; those are printed on
    completion so the audience sees the outcome next to the duration.
    """
    est = f"  [dim](estimate: {estimate})[/dim]" if (estimate and _HAS_RICH) else (
        f"  (estimate: {estimate})" if estimate else "")
    _emit(f"  [cyan]>[/cyan] {label} ...{est}" if _HAS_RICH else f"  > {label} ...{est}")
    t0 = time.perf_counter()
    payload: dict[str, Any] = {}
    try:
        yield payload
    finally:
        dt = time.perf_counter() - t0
        _TIMINGS.append({"stage": label, "seconds": round(dt, 3)})
        extra = ""
        if payload:
            extra = "  " + "  ".join(f"{k}={v}" for k, v in payload.items())
        if _HAS_RICH:
            _emit(f"  [green]done[/green] {label}{extra}  "
                  f"[bold]{_fmt_secs(dt)}[/bold]")
        else:                               # pragma: no cover
            _emit(f"  done {label}{extra}  {_fmt_secs(dt)}")


class StageTimer:
    """Wall-clock timer for a whole pipeline stage."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.t0 = time.perf_counter()

    def finish(self, extra: str = "") -> float:
        dt = time.perf_counter() - self.t0
        _TIMINGS.append({"stage": f"TOTAL {self.name}", "seconds": round(dt, 3)})
        blank()
        if _console is not None and _unicode_ok():
            body = Text(f"{self.name} complete", style="bold green")
            body.append(f"\nelapsed: {_fmt_secs(dt)}", style="white")
            if extra:
                body.append(f"\n{extra}", style="dim")
            _console.print(Panel(body, border_style="green", padding=(0, 2)))
        else:                               # pragma: no cover
            print(f"[DONE] {self.name}  elapsed {_fmt_secs(dt)}  {extra}")
        return dt


def timings() -> list[dict[str, Any]]:
    return list(_TIMINGS)


#: Timings accumulate across runs so a stage can be compared with itself over
#: time, but the file is a diagnostic rather than a deliverable - cap it so it
#: cannot grow without bound on a machine that runs the pipeline repeatedly.
MAX_TIMING_ENTRIES = 400


def dump_timings(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with contextlib.suppress(Exception):
            existing = json.loads(path.read_text(encoding="utf-8"))
    combined = (existing + _TIMINGS)[-MAX_TIMING_ENTRIES:]
    path.write_text(json.dumps(combined, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def table(title: str, columns: list[str], rows: list[list[Any]],
          highlight_first_col: bool = True) -> None:
    """Print a results table a reader can take in at a glance."""
    numeric_col = [
        all(isinstance(r[i], (int, float)) or _looks_numeric(r[i]) for r in rows)
        if rows else False
        for i in range(len(columns))
    ]
    if _console is not None and _unicode_ok():
        t = Table(title=title, title_style="bold", header_style="bold cyan",
                  show_lines=False, pad_edge=False)
        for i, c in enumerate(columns):
            t.add_column(str(c), justify="right" if numeric_col[i] else "left",
                         style="bold" if (i == 0 and highlight_first_col) else None,
                         no_wrap=(i == 0))
        for r in rows:
            t.add_row(*[_cell(x) for x in r])
        _console.print(t)
    else:                                   # pragma: no cover
        print(f"\n{title}")
        widths = [max(len(str(c)), *(len(_cell(r[i])) for r in rows)) if rows
                  else len(str(c)) for i, c in enumerate(columns)]
        print("  ".join(str(c).ljust(w) for c, w in zip(columns, widths)))
        print("  ".join("-" * w for w in widths))
        for r in rows:
            print("  ".join(_cell(x).ljust(w) for x, w in zip(r, widths)))


def _cell(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _looks_numeric(x: Any) -> bool:
    try:
        float(str(x).replace(",", "").replace("%", "").strip())
        return True
    except ValueError:
        return False


def kv(pairs: dict[str, Any], title: str = "") -> None:
    """Print a small key/value block."""
    if title:
        _emit(f"  [bold]{title}[/bold]" if _HAS_RICH else f"  {title}")
    width = max((len(k) for k in pairs), default=0)
    for k, v in pairs.items():
        val = _cell(v)
        _emit(f"    [dim]{k.ljust(width)}[/dim]  {val}" if _HAS_RICH
              else f"    {k.ljust(width)}  {val}")
