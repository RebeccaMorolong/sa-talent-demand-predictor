from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    level="INFO",
)
logger.add(
    "logs/pipeline.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)
