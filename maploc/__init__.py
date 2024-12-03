import logging
import coloredlogs
from pathlib import Path

logger = logging.getLogger("MapLoc")
coloredlogs.install(
    level="DEBUG",
    logger=logger,
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
)

repo_dir = Path(__file__).resolve().parent.parent
CONFIG_PATH = repo_dir / "maploc/conf"
