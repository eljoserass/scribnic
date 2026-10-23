"""Optional execution notifications, scoped to a caller and independent of any UI."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal
from uuid import uuid4


@dataclass(frozen=True)
class ProgressEvent:
    """Describe one phase without retaining inputs, generated text or UI resources."""

    task_id: str
    label: str
    state: Literal["start", "update", "done", "error", "interrupted"]
    detail: str = ""
    completed: int = 0
    total: int | None = None
    unit: Literal["bytes", "tokens"] | None = None


ProgressObserver = Callable[[ProgressEvent], None]
_observer: ContextVar[ProgressObserver | None] = ContextVar(
    "scribnic_progress_observer", default=None
)


def current_observer() -> ProgressObserver | None:
    """Return the optional observer belonging to the current execution context."""
    return _observer.get()


@contextmanager
def observe_progress(observer: ProgressObserver | None) -> Iterator[None]:
    """Install a caller-owned observer temporarily and restore it on every exit."""
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


class ProgressTask:
    """Send notifications for one phase; own no models, files or terminal state."""

    def __init__(
        self,
        label: str,
        *,
        detail: str = "",
        observer: ProgressObserver | None = None,
        total: int | None = None,
        unit: Literal["bytes", "tokens"] | None = None,
    ) -> None:
        """Start a phase using an explicit observer or the current caller's observer."""
        self.observer = observer if observer is not None else current_observer()
        self.event = ProgressEvent(uuid4().hex, label, "start", detail, 0, total, unit)
        self._send()

    def _send(self) -> None:
        """Deliver the current immutable event when presentation was requested."""
        if self.observer is not None:
            self.observer(self.event)

    def update(self, completed: int) -> None:
        """Report an absolute byte or token count without assuming a final total."""
        self.event = replace(self.event, state="update", completed=completed)
        self._send()

    def finish(self, state: Literal["done", "error", "interrupted"] = "done") -> None:
        """Finish a phase while preserving its actual measured progress."""
        self.event = replace(self.event, state=state)
        self._send()


@contextmanager
def phase(label: str, *, detail: str = "") -> Iterator[ProgressTask]:
    """Notify phase completion, failure or cancellation without swallowing errors."""
    task = ProgressTask(label, detail=detail)
    try:
        yield task
    except BaseException as error:
        task.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "error")
        raise
    else:
        task.finish()
