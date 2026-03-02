from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %I:%M:%S %p %Z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        extras: dict[str, Any] = getattr(record, "extras", {})
        payload.update(extras)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(
    debug: bool, log_level: str = "INFO", log_json: bool = True
) -> None:
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if debug or not log_json:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
    else:
        handler.setFormatter(JsonFormatter())

    root.addHandler(handler)
    level = (
        logging.DEBUG if debug else getattr(logging, log_level.upper(), logging.INFO)
    )
    root.setLevel(level)
