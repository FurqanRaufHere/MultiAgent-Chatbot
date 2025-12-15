from loguru import logger
import sys

def setup_logger():
    # Configure the logger to output to stderr and remove the default handler
    logger.remove()
    
    # Add a custom handler for clear, structured logging
    logger.add(
        sys.stderr, 
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    return logger

# Global logger instance used across the project
log = setup_logger()