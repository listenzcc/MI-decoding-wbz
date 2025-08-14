from loguru import logger

logger.add('log/debug.log', level='DEBUG', rotation='5 MB')
logger.add('log/info.log', level='INFO', rotation='5 MB')
