"""
main.py
Entry point for the arXiv crawler.

Usage
-----
    python -m backend.crawler.main
    python -m backend.crawler.main --batch-size 5 --discovery-interval 60
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (must happen before local imports that use loggers)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).resolve().parent.parent / "data" / "crawler.log",
            encoding="utf-8",
        ),
    ],
)

from .crawler import Crawler, CrawlerConfig  # noqa: E402  (after logging setup)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="arXiv AI/ML crawler — discovers and stores article metadata."
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Documents to download per cycle (default: 10).",
    )
    p.add_argument(
        "--ids-per-discovery",
        type=int,
        default=100,
        metavar="N",
        help="IDs to fetch from arXiv per discovery cycle (default: 100).",
    )
    p.add_argument(
        "--discovery-interval",
        type=float,
        default=120.0,
        metavar="SECS",
        help="Seconds between discovery cycles (default: 120).",
    )
    p.add_argument(
        "--download-interval",
        type=float,
        default=30.0,
        metavar="SECS",
        help="Seconds between download cycles (default: 30).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Ensure the data directory exists
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    config = CrawlerConfig(
        ids_per_discovery=args.ids_per_discovery,
        batch_size=args.batch_size,
        discovery_interval=args.discovery_interval,
        download_interval=args.download_interval,
    )

    crawler = Crawler(config=config)
    crawler.run_forever()


if __name__ == "__main__":
    main()