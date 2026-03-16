"""
main.py
Arranca el crawler completo (discovery + metadatos + PDFs → SQLite).

Uso
---
    python -m backend.main
    python -m backend.main --batch-size 5 --pdf-batch 3
"""
from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "app.log", encoding="utf-8"),
    ],
)

from crawler import Crawler, CrawlerConfig


def parse_args():
    p = argparse.ArgumentParser(description="OmniRetrieve — arXiv crawler")
    p.add_argument("--batch-size",         type=int,   default=10,    help="Metadatos a descargar por ciclo")
    p.add_argument("--pdf-batch",          type=int,   default=5,     help="PDFs a descargar por ciclo")
    p.add_argument("--ids-per-discovery",  type=int,   default=100,   help="IDs por ciclo de discovery")
    p.add_argument("--discovery-interval", type=float, default=120.0, help="Segundos entre ciclos de discovery")
    p.add_argument("--download-interval",  type=float, default=30.0,  help="Segundos entre ciclos de metadatos")
    p.add_argument("--pdf-interval",       type=float, default=60.0,  help="Segundos entre ciclos de PDF")
    return p.parse_args()


def main():
    args = parse_args()
    crawler = Crawler(config=CrawlerConfig(
        ids_per_discovery  = args.ids_per_discovery,
        batch_size         = args.batch_size,
        pdf_batch_size     = args.pdf_batch,
        discovery_interval = args.discovery_interval,
        download_interval  = args.download_interval,
        pdf_interval       = args.pdf_interval,
    ))
    crawler.run_forever()

if __name__ == "__main__":
    main()