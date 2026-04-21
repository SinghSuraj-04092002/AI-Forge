from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from multiagent_sds.models.domain import ProjectContext, ProjectStatus


class StateStore:
    def __init__(self) -> None:
        self._store: Dict[str, ProjectContext] = {}
        self._lock = asyncio.Lock()

    async def save(self, ctx: ProjectContext) -> None:
        async with self._lock:
            self._store[ctx.project_id] = ctx

    async def get(self, project_id: str) -> Optional[ProjectContext]:
        async with self._lock:
            return self._store.get(project_id)

    async def list_all(self) -> List[ProjectContext]:
        async with self._lock:
            return list(self._store.values())

    async def delete(self, project_id: str) -> bool:
        async with self._lock:
            if project_id in self._store:
                del self._store[project_id]
                return True
            return False

    def __len__(self) -> int:
        return len(self._store)


store = StateStore()
