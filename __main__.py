import logging
from gui import GUI
from network import HopfieldNetwork
logger = logging.getLogger(__name__)
logging.basicConfig()
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    logger.info("starting")
    gui = GUI()
    logger.info("exiting.")
