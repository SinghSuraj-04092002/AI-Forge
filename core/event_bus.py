from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, project_id: str, event_type: str, data: Any) -> None:
        payload = {
            "type": event_type,
            "project_id": project_id,
            "ts": datetime.utcnow().isoformat(),
            "data": data,
        }
        async with self._lock:
            queues = list(self._subscribers.get(project_id, []))
        for q in queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # slow consumer – drop rather than block

    async def subscribe(self, project_id: str) -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        async with self._lock:
            self._subscribers[project_id].append(q)
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            async with self._lock:
                try:
                    self._subscribers[project_id].remove(q)
                except ValueError:
                    pass

    async def unsubscribe_all(self, project_id: str) -> None:
        async with self._lock:
            self._subscribers.pop(project_id, None)


bus = EventBus()
