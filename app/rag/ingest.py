"""CLI de ingesta:  python -m app.rag.ingest [--reset]"""
import logging
import sys

from app.rag.store import ingest

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ingest(reset="--reset" in sys.argv)
