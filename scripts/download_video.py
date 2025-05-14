import sys
from configargparse import ArgumentParser
from pathlib import Path
from categories import category_ids, process_category
from models.youtube_downloader import YoutubeDownloader
from models.utils import Logger

from database import parse_version


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, is_config_file=True, help="Specify a config file path")
    parser.add_argument("--base_path", type=str, help="Base path to the dataset")
    parser.add_argument("--db_path", type=str, help="Path to database file")
    parser.add_argument("--max_download", type=int, default=None, help="Max number of videos to download")
    parser.add_argument("--max_query", type=int, default=None, help="Max number of queries to generate")
    parser.add_argument("--max_video_per_category", type=int, default=None, help="Max number of videos to download per category")
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--version", type=str, default=None, help="Version of the dataset")
    parser.add_argument(
        "--dry_run", action="store_true", help="Scrape and insert rows to dataset without downloading"
    )
    parser.add_argument(
        "--categories", type=str, nargs='+', default=None,
        help="Category ids or names to download, None to download all categories listed in categories/__init__.py"
    )
    parser.add_argument(
        "--filter_words", type=str, nargs='+', default=None,
        help="words to filter for generating queries, i.e. music, song"
    )
    parser.add_argument(
        "--query_word_method", type=str, default="gpt", choices=["wordnet", "gpt"],
        help="Method to get query word expansion of a search word"
    )
    parser.add_argument(
        "--query_phrase_method", type=str, default="gpt", choices=["pytube", "selenium", "gpt"],
        help="Method to get query phrases of a search word"
    )
    parser.add_argument(
        "--scrape_method", type=str, default="selenium", choices=["selenium", "pytube"],
        help="Method to scrape youtube video ids"
    )
    parser.add_argument(
        "--num_processes", type=int, default=10,
        help="Number of processes used in multiprocess downloading"
    )
    parsed_args, _ = parser.parse_known_args()
    return parsed_args


if __name__ == "__main__":
    args = parse_args()
    # TODO: maybe move logger inside class
    video_version = parse_version(args.version).get("video")
    video_dir = f"video_{video_version}" if video_version else "video"
    sys.stdout = Logger(Path(args.base_path) / video_dir / "log.txt")
    if args.categories is None:  # download all categories
        args.categories = list(category_ids.keys())
    for cat in args.categories:
        _, animal = process_category(cat)
        downloader = YoutubeDownloader(
            download_dir=Path(args.base_path) / video_dir / animal,
            db_path=args.db_path,
            category=animal,
            query_word_method=args.query_word_method,
            query_phrase_method=args.query_phrase_method,
            filter_words=args.filter_words,
            scrape_method=args.scrape_method,
            max_download=args.max_download,
            max_query=args.max_query,
            max_video_per_category=args.max_video_per_category,
            num_processes=args.num_processes,
            verbose=args.verbose,
            dry_run=args.dry_run,
            version=args.version
        )
        video_paths = downloader.run()
        print(f"Finished downloading {animal}")
