"""OmniBoxAgent application entry point."""

import logging
import sys

import uvicorn

from omnibox_agent.core.config import get_config
from omnibox_agent.core.tracing import install_trace_filter, create_trace_formatter

install_trace_filter()

fmt = "%(asctime)s [%(levelname)s] [%(trace_id)s] %(name)s: %(message)s"
formatter = create_trace_formatter(fmt)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

log = logging.getLogger(__name__)


def main():
    cfg = get_config()
    log.info("Starting OmniBoxAgent on %s:%s", cfg.agent.host, cfg.agent.port)
    uvicorn.run(
        "omnibox_agent.api.app:app",
        host=cfg.agent.host,
        port=cfg.agent.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
