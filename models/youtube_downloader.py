import cv2
import json
import os
import re
import nltk
import random
import traceback
from tqdm import tqdm
from time import time, sleep
from ast import literal_eval
from itertools import product, repeat
from openai import AzureOpenAI, OpenAI
from dataclasses import dataclass
from nltk.corpus import wordnet as wn
from multiprocessing import Pool, Value
from pytubefix import YouTube, Search
from pytubefix.cli import on_progress
from pytubefix.innertube import InnerTube
from pytubefix.exceptions import AgeRestrictedError
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

from database import Database, Status


class YoutubeDownloader:
    """Downloader class for querying, scraping, and downloading videos given a search word"""
    def __init__(
        self, download_dir, db_path, category, verbose=True, max_retry=3, shuffle=True,
        query_word_method="gpt", query_phrase_method="gpt", filter_words=None, scrape_method="selenium",
        max_video_per_category=None, num_processes=10, max_query=10, max_download=100, dry_run=False, version=None
    ):
        # General attributes
        self.category = category
        self.download_dir = download_dir
        self.max_retry = max_retry
        self.verbose = verbose
        self.dry_run = dry_run
        self.db = Database(db_path=db_path, version=version)
        # Query attributes
        self.max_query = max_query
        self.query_word_method = query_word_method
        self.query_phrase_method = query_phrase_method
        self.filter_words = [w.lower() for w in filter_words]
        # self.gpt_client = AzureOpenAI(
        #     azure_endpoint=os.environ.get("OPENAI_API_BASE"),
        #     api_key=os.environ.get("OPENAI_API_KEY"),
        #     api_version="2023-07-01-preview",
        #     max_retries=max_retry
        # )
        self.gpt_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_retries=max_retry
        )
        self.query_word_prompt = (
            "List {num} types of {category}. Only show the list in python list format without using a code block. "
        )
        self.query_phrase_prompt = (
            "List {num} search phrases or autocompletions for searching {category} videos on a video sharing website. "
            "Assume user already input the word {category}, only show the trailing phrases. "
            "Only show the list in python list format without using a code block. "
        )
        # Scrape attributes
        self.scrape_method = scrape_method
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.selenium_driver = Chrome(options=options)
        self.wait = WebDriverWait(self.selenium_driver, 180)
        self.video_url_patterns = [  # Only work for scrape_selenium
            "youtube.com/watch?v=",  # search for video
            "youtube.com/shorts/"  # search for short
        ]
        # Download attributes
        self.shuffle = shuffle
        self.max_video_per_category = max_video_per_category
        self.max_download = max_download
        self.num_processes = num_processes
        self.counter = Value('i', 0)
        self.init_database()
        os.makedirs(self.download_dir, exist_ok=True)
        if self.verbose:
            print(f"Downloading video to {self.download_dir} using {self.num_processes} processes")

    def init_database(self):
        self.db.make_video_table()

    def query_gpt(self, text):
        retry = 0
        messages = [{"role": "user", "content": text}]
        while True:
            if retry > self.max_retry:
                return None
            try:
                response = self.gpt_client.chat.completions.create(
                    messages=messages,
                    model="gpt-4",
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                ).choices[0].message.content
                return response
            except Exception as e:
                print(f"{type(e).__name__}: {str(e)}", flush=True)
                sleep(2)
                retry += 1
                continue

    def generate_query_words(self, wn_max_depth=2, max_num=30):
        """
        From category generate a list of [max_num] related words by
        - retrieving a wordnet closure (a set of children classes) with [wn_max_depth] or
        - querying ChatGPT
        """
        query_words = []
        if self.query_word_method == "wordnet":
            nltk.download("wordnet", quiet=True)
            synsets = wn.synsets(self.category)
            for s in synsets:
                closure = s.closure(lambda x: x.hyponyms(), depth=wn_max_depth)
                for h in closure:
                    if h.pos() == wn.NOUN:
                        for name in h.lemma_names():
                            name = name.replace('_', ' ').strip()
                            if self.category not in name:
                                name = f"{name} {self.category}"
                            query_words.append(name)
        elif self.query_word_method == "gpt":
            prompt = self.query_word_prompt.format(num=max_num, category=self.category)
            response = self.query_gpt(prompt)
            try:
                query_words += literal_eval(response)
            except (ValueError, SyntaxError) as e:
                print(f"{type(e).__name__}: {str(e)}", flush=True)
                self.query_word_method = "wordnet"
                query_words = self.generate_query_words()
        else:
            raise NotImplementedError
        if max_num:
            query_words = query_words[:max_num]
        query_words = [
            q.strip() if self.category.lower() in q.lower()
            else f"{q.strip()} {self.category}"
            for q in query_words
        ]
        if self.verbose:
            print(f"Generated {len(query_words)} query words: {query_words}", flush=True)
        return query_words

    def generate_query_phrases(self, max_num=30, shuffle=True):
        """
        From category generate a list of [max_num] related search phrases by
        - retrieving YouTube autocompletion (by pytube or selenium automation) or
        - querying ChatGPT
        Output does not include the input word at beginning
        """
        query_phrases = None
        if self.query_phrase_method == "pytube":
            query_phrases = SearchAutocompletion(self.category + ' ').completion_suggestions
        elif self.query_phrase_method == "selenium":
            self.selenium_driver.get("https://www.youtube.com")
            self.wait.until(EC.presence_of_element_located((By.NAME, "search_query"))).send_keys(self.category + ' ')
            retry = 0
            while True:
                if retry > self.max_retry:
                    self.query_phrase_method = "pytube"
                    query_phrases = self.generate_query_phrases()
                    break
                ActionChains(self.selenium_driver).send_keys(Keys.ARROW_DOWN).perform()
                try:
                    self.selenium_driver.find_element(By.CLASS_NAME, "sbqs_c")
                    ac_list = self.selenium_driver.find_elements(By.CLASS_NAME, "sbqs_c")
                    query_phrases = [ac.text.replace(self.category, '').strip() for ac in ac_list]
                    break
                except NoSuchElementException as e:
                    print(f"{type(e).__name__}: {str(e)}", flush=True)
                    sleep(1)
                    retry += 1
                    continue
        elif self.query_phrase_method == "gpt":
            prompt = self.query_phrase_prompt.format(num=max_num, category=self.category)
            response = self.query_gpt(prompt)
            try:
                query_phrases = literal_eval(response)
            except (ValueError, SyntaxError) as e:
                print(f"{type(e).__name__}: {str(e)}", flush=True)
                self.query_phrase_method = "selenium"
                query_phrases = self.generate_query_phrases()
        else:
            raise NotImplementedError
        if shuffle:
            random.shuffle(query_phrases)
        if max_num:
            query_phrases = query_phrases[:max_num]
        query_phrases = [
            q.strip() if q.strip().split()[0] != self.category
            else ' '.join(q.strip().split()[1:])
            for q in query_phrases
        ]
        if self.verbose:
            print(f"Generated {len(query_phrases)} query phrases for {self.category}: ", flush=True)
            for i, q in enumerate(query_phrases):
                print(f"\t{i + 1}. {q}", flush=True)
        return query_phrases

    def generate_search_queries(self, shuffle=True):
        """
        Input a word, find related query words and query phrase
        Combine words and phrases into [max_num] search queries
        Filter all queries that contain any of the words in filter_words
        """
        query_words = self.generate_query_words()
        query_phrases = self.generate_query_phrases()
        queries = [f"{h.lower()} {p.lower()}" for (h, p) in product(query_words, query_phrases)]
        queries = list(set(queries))
        if self.filter_words:
            for w in self.filter_words:
                queries = [q for q in queries if w not in q.split()]
        if shuffle:
            random.shuffle(queries)
        if self.max_query:
            queries = queries[:self.max_query]
        if self.verbose:
            print(f"Generated {len(queries)} queries for {self.category}: ", flush=True)
            for i, q in enumerate(queries):
                print(f"\t{i + 1}. {q}", flush=True)
        return queries

    @staticmethod
    def scrape_pytube(query):
        """Deprecated"""
        results = Search(query).results
        video_ids = [r.video_id for r in results]
        return video_ids

    def scrape_selenium(self, query):
        def get_id_from_url(url):
            if not url:
                return None
            id = None
            for pattern in self.video_url_patterns:
                idx = url.find(pattern)
                if idx >= 0:
                    id = url[idx + len(pattern):]
                    break
            if id is not None:
                for i, c in enumerate(id):
                    if c == '&':
                        return id[:i]
            return id
        video_ids = []
        self.selenium_driver.get(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
        self.wait.until(EC.presence_of_all_elements_located((By.ID, "thumbnail")))
        elements = self.selenium_driver.find_elements(By.TAG_NAME, 'a')
        for element in elements:
            retry = 0
            url = None
            while retry < self.max_retry:
                try:
                    url = element.get_attribute("href")
                    break
                except StaleElementReferenceException:
                    retry += 1
                    sleep(2)
            if url is None:
                continue
            id = get_id_from_url(url)
            if id is not None:
                video_ids.append(id)
        return video_ids

    def scrape(self, query):
        if self.scrape_method == "selenium":
            video_ids = self.scrape_selenium(query)
        elif self.scrape_method == "pytube":
            video_ids = self.scrape_pytube(query)
        else:
            raise NotImplementedError
        video_ids = list(set(video_ids))
        if self.verbose:
            print(f"Searching: {query}, found {len(video_ids)} videos")
        return video_ids

    @staticmethod
    def _download(video_id, query, download_dir, filter_words=None, verbose=False, dry_run=False):
        """
        Static method for multiprocessing download,
        downloaded video at download_dir/video_id/video_id.mp4
        """
        result = DownloadResult(
            video_id=video_id,
            query_text=query,
        )
        try:
            yt = YouTubeBypassAge(
                "https://youtube.com/watch?v="+video_id, on_progress_callback=on_progress, client="WEB"
            )
            streams = yt.streams.filter(
                type="video", file_extension="mp4", progressive=False,
                # TODO: check why av01 codec not readable by opencv ffmpeg
                custom_filter_functions=[lambda s: not s.video_codec.startswith("av01")]
            ).order_by("resolution")
            stream = streams.last()
            num_frames = int(stream.fps * yt.length)
            download_path = os.path.join(download_dir, f"{video_id}.mp4")
            result.fps = stream.fps
            result.duration = yt.length  # Duration and fps are rounded and not accurate
            result.frames = num_frames
            result.title = yt.title
            title_words = re.sub(r'[^a-zA-Z]', ' ', yt.title).lower().split()
            if filter_words:
                for w in filter_words:
                    if w in title_words:
                        raise ValueError(f"Title contains filter word {w}")
            result.keywords = json.dumps(yt.keywords)
            os.makedirs(download_dir, exist_ok=True)
            if verbose:
                print(
                    f"Downloading {video_id}: duration {yt.length}s (~{num_frames} frames), "
                    f"codec {stream.video_codec}, resolution {stream.resolution}",
                    flush=True
                )
            start = time()
            if not dry_run:
                stream.download(
                    output_path=download_dir,
                    filename=f"{video_id}.mp4",
                    filename_prefix=None,
                    skip_existing=True,
                    timeout=None,
                    max_retries=3,
                )
                result.download_path = download_path
                print("Download path: ", download_path, flush=True)
                cap = cv2.VideoCapture(download_path)
                result.frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                result.fps = cap.get(cv2.CAP_PROP_FPS)
                result.duration = result.frames / result.fps
                cap.release()
            result.download_time = time() - start
            result.success = True
        except Exception as e:
            result.success = False
            result.reason = f"{type(e).__name__}: {str(e)}"
        return result

    def filter_existing_video(self, video_ids):
        """Filter all videos that are already accessed/downloaded/discarded in database"""
        filtered_video_ids = []
        for video_id in video_ids:
            status = self.db.get_video_status(video_id=video_id)
            if status is not None:
                print(f"{video_id} already accessed with status {status}", flush=True)
            else:
                filtered_video_ids.append(video_id)
        return filtered_video_ids

    def download(self, video_id_to_query):
        """Download video given video id with multiprocessing"""
        self.counter.value = 0
        pool = Pool(self.num_processes)
        video_ids = list(video_id_to_query.keys())
        video_ids = self.filter_existing_video(video_ids)
        if self.shuffle:
            random.shuffle(video_ids)
        if self.max_download:
            video_ids = video_ids[:self.max_download]
        if self.max_video_per_category:
            existing_videos = len(self.get_existing_video())
            if existing_videos >= self.max_video_per_category:
                video_ids = []
            else:
                video_ids = video_ids[:self.max_video_per_category - existing_videos]
            if self.verbose:
                print(
                    f"{existing_videos} videos already downloaded for {self.category}, "
                    f"downloading {len(video_ids)} more videos", flush=True
                )
        queries = [video_id_to_query[video_id] for video_id in video_ids]
        results = []
        args = zip(
            video_ids, queries, repeat(self.download_dir), repeat(self.filter_words),
            repeat(self.verbose), repeat(self.dry_run)
        )
        callback = lambda result: self.on_complete(
            result, self.category, counter=self.counter, total=len(video_ids), db=self.db
        )
        # yt = YouTubeBypassAge("https://youtube.com/watch?v=" + video_ids[0]).streams  # prompt for auth
        for arg in args:
            result = pool.apply_async(self._download, arg, callback=callback)
            results.append(result)
        pool.close()
        pool.join()
        if self.verbose:
            print(f"\nDownloaded {self.counter.value}/{len(video_ids)} videos", flush=True)
        video_paths = [result.get().download_path for result in results if result.get().success]
        return video_paths

    @staticmethod
    def on_complete(result, category, counter, total, db):
        """Static callback to update download progress and database"""
        if result.success:
            with counter.get_lock():
                counter.value += 1
                print(
                    f"Downloaded {result.video_id} ({counter.value}/{total}) in {result.download_time:.2f}s",
                    flush=True
                )
            db.insert_video(
                result.video_id, category, video_path=result.download_path, duration=result.duration, fps=result.fps,
                frames=result.frames, title=result.title, keywords=result.keywords, query_text=result.query_text
            )
        else:
            # db.insert_video(
            #     result.video_id, category, reason=result.reason, duration=result.duration, fps=result.fps,
            #     frames=result.frames, title=result.title, keywords=result.keywords, query_text=result.query_text
            # )
            print(f"Unable to download {result.video_id} due to {result.reason}", flush=True)

    def get_existing_video(self):
        """Get all existing videos of self.category in database"""
        condition = (
            f"category='{self.category}' and "
            f"(status='{Status.DOWNLOADED}' or status='{Status.PROCESSING}' or status='{Status.PROCESSED}')"
        )
        videos = self.db.get_all_videos(condition=condition)
        return videos

    def run(self):
        """Run generate queries, scrape, and download"""
        if self.max_video_per_category:
            existing_videos = len(self.get_existing_video())
            if existing_videos >= self.max_video_per_category:
                print(f"{existing_videos} videos already downloaded for {self.category}, skipping", flush=True)
                return []
        queries = self.generate_search_queries()
        video_id_to_query = {}
        for q in tqdm(queries):
            try:
                video_ids = self.scrape(q)
                for video_id in video_ids:
                    video_id_to_query[video_id] = q
            except Exception as e:
                print(f"Error scraping {q}: {e}", flush=True)
                traceback.print_exc()
        video_paths = self.download(video_id_to_query)
        return video_paths


class YouTubeBypassAge(YouTube):
    """Custom YouTube stream class that bypass the age restriction when accessing videos"""
    def bypass_age_gate(self):
        innertube = InnerTube(
            client="ANDROID_CREATOR",
            use_oauth=self.use_oauth,
            allow_cache=self.allow_oauth_cache
        )
        innertube_response = innertube.player(self.video_id)
        playability_status = innertube_response['playabilityStatus'].get('status', None)
        if playability_status == 'UNPLAYABLE':
            raise AgeRestrictedError(self.video_id)
        self._vid_info = innertube_response


class SearchAutocompletion(Search):
    """Custom pytube Search class that skips parsing search results and allows different clients"""
    def __init__(self, query, client="WEB"):
        super().__init__(query)
        assert client in ["WEB", "IOS"]
        self.client = client
        self._innertube_client = InnerTube(client=self.client)

    def fetch_and_parse(self, continuation=None):
        raw_results = self.fetch_query(continuation)
        self._completion_suggestions = raw_results["refinements"]
        return None, None


@dataclass
class DownloadResult:
    """Multiprocess pickleable data class for storing and download result information"""
    video_id: str
    query_text: str
    duration: float = None
    fps: float = None
    frames: int = None
    title: str = None
    keywords: str = None
    download_path: str = None
    success: bool = None
    reason: str = None
