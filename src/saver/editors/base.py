"""Base interfaces for editor adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from saver.types import EditorProposal, ProbeBundle


class BaseEditorAdapter(ABC):
    """Interface for editors that support propose/commit semantics."""

    @abstractmethod
    def propose(self, probe_bundle: ProbeBundle) -> EditorProposal:
        """Return a candidate edit without committing it permanently."""

    @abstractmethod
    def commit(self, proposal: EditorProposal) -> None:
        """Commit a previously proposed edit."""

    @abstractmethod
    def rollback(self, proposal: EditorProposal) -> None:
        """Discard a previously proposed edit."""
