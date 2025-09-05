"""Lightweight package initializer for Elysia.

This keeps top-level imports minimal to avoid pulling in optional/heavy
dependencies (e.g., pydantic, dspy) during basic usage like importing
`elysia.api.*` or `elysia.objects` in test environments without extras.

Downstream code should import submodules directly, e.g.:
    from elysia.api.agent_manager import AgentManager
    from elysia.objects import Tool, Result

Power users can still import advanced modules explicitly, e.g.:
    from elysia.tree.tree import Tree
    from elysia.config import settings
"""

from elysia.__metadata__ import (
    __version__,
    __name__,
    __description__,
    __url__,
    __author__,
    __author_email__,
)

# Intentionally avoid importing heavy submodules at package import time.
# Import these modules directly where needed instead of re-exporting here.
