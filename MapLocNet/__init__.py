import logging
import coloredlogs


logger = logging.getLogger("MapLocNet")
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
)
