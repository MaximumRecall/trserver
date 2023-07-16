import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any, Optional
from urllib.parse import urlparse
from uuid import uuid4, uuid1

from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args


# data model:
# we have urls, paths, and chunks.
# the url table records the full url.  Every time we save a page, we save a new row here.
# the paths table records url hostname and path.  Before saving a page, we check the most
# recent version of the page in the paths table.  If it's the same, we don't save the page.
# This allows us to accommodate urls that only differ by query string, without saving
# multiple copies of the same page.
class DB:
    def __init__(self, cluster: Cluster) -> None:
        self.keyspace = "total_recall"
        self.table_chunks = "saved_chunks"
        self.table_urls = "saved_urls"
        self.table_paths = "saved_paths"
        self.cluster = cluster
        self.session = self.cluster.connect()

        # Keyspace (don't try to create unless it's a local cluster)
        if self.cluster.contact_points == ['127.0.0.1']:
            self.session.execute(
                f"""
                CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
                WITH REPLICATION = {{ 'class': 'SimpleStrategy', 'replication_factor': 1 }}
                """
            )

        # URLs table
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table_urls} (
            user_id uuid,
            url_id timeuuid,
            full_url text,
            title text,
            PRIMARY KEY (user_id, url_id));
            """
        )
        url_index_name = f"{self.table_urls}_full_url_idx"
        self.session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {url_index_name} ON {self.keyspace}.{self.table_urls} (full_url)
            USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
            """
        )

        # Paths table
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table_paths} (
            user_id uuid,
            path text,
            url_id timeuuid,
            text_content text,
            formatted_content text,
            PRIMARY KEY ((user_id, path), url_id))
            """
        )

        # Chunks table
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table_chunks} (
            user_id uuid,
            url_id timeuuid,
            full_url text,
            title text,
            chunk text,
            embedding_e5 vector<float, 384>,
            PRIMARY KEY (user_id, chunk));
            """
        )
        # Embedding index
        embedding_index_name = f"{self.table_chunks}_embedding_idx"
        self.session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {embedding_index_name} ON {self.keyspace}.{self.table_chunks} (embedding_e5)
            USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
            WITH OPTIONS = {{ 'similarity_function': 'dot_product' }}
            """
        )

    def last_version(self, user_id: uuid4, path: str) -> Optional[str]:
        st = self.session.prepare(
            f"""
            SELECT text_content
            FROM {self.keyspace}.{self.table_paths}
            WHERE user_id = ? AND path = ?
            ORDER BY url_id DESC LIMIT 1
            """
        )
        row = self.session.execute(st, (user_id, path)).one()
        return row and row.text_content

    def upsert_chunks(self,
                      user_id: uuid4,
                      path: str,
                      full_url: str,
                      title: str,
                      text_content: str,
                      chunks: List[Tuple[str, List[float]]]) -> None:
        st_urls = self.session.prepare(
            f"""
            INSERT INTO {self.keyspace}.{self.table_urls}
            (user_id, url_id, full_url, title)
            VALUES (?, ?, ?, ?)
            """
        )
        st_paths = self.session.prepare(
            f"""
            INSERT INTO {self.keyspace}.{self.table_paths}
            (user_id, path, url_id, text_content)
            VALUES (?, ?, ?, ?)
            """
        )
        # TODO retry?
        url_uuid = uuid1()
        self.session.execute(st_urls, (user_id, url_uuid, full_url, title))
        self.session.execute(st_paths, (user_id, path, url_uuid, text_content))

        st_chunks = self.session.prepare(
            f"""
            INSERT INTO {self.keyspace}.{self.table_chunks}
            (user_id, url_id, full_url, title, chunk, embedding_e5)
            VALUES (?, ?, ?, ?, ?, ?)
            """
        )
        denormalized_chunks = [(user_id, url_uuid, full_url, title, chunk, embedding)
                               for chunk, embedding in chunks]
        backoff = 0.5
        while denormalized_chunks and backoff < 60:
            results = execute_concurrent_with_args(self.session, st_chunks, denormalized_chunks,
                                                   concurrency=16, raise_on_first_error=False)
            denormalized_chunks = [chunk for chunk, (success, _)
                                   in zip(denormalized_chunks, results) if not success]
            time.sleep(backoff)
            backoff *= 2
        if denormalized_chunks:
            raise Exception(f"Failed to insert {len(denormalized_chunks)} chunks")


    def recent_urls(self, user_id: uuid4, saved_before: Optional[datetime], limit: int) -> List[Dict[str, Union[str, datetime]]]:
        if saved_before:
            cql = f"""
                  SELECT full_url, title, url_id 
                  FROM {self.keyspace}.{self.table_urls} 
                  WHERE user_id = ? AND url_id < minTimeuuid(?)
                  ORDER BY url_id DESC
                  LIMIT ?
                  """
        else:
            cql = f"""
                  SELECT full_url, title, url_id 
                  FROM {self.keyspace}.{self.table_urls} 
                  WHERE user_id = ? 
                  ORDER BY url_id DESC
                  LIMIT ?
                  """
        query = self.session.prepare(
            cql
        )
        if saved_before:
            results = self.session.execute(query, (user_id, saved_before, limit))
        else:
            results = self.session.execute(query, (user_id, limit))
        return [{k: getattr(row, k) for k in ['full_url', 'title', 'url_id']} for row in results]


    def search(self, user_id: uuid4, vector: List[float]) -> List[Dict[str, Union[Tuple[str, float], datetime, List[str]]]]:
        query = self.session.prepare(
            f"""
            SELECT full_url, title, chunk, url_id, similarity_dot_product(embedding_e5, ?) as score
            FROM {self.keyspace}.{self.table_chunks} 
            WHERE user_id = ? 
            ORDER BY embedding_e5 ANN OF ? LIMIT 10
            """
        )
        result_set = self.session.execute(query, (vector, user_id, vector))
        url_dict = defaultdict(lambda: {'chunks': [], 'title': None, 'url_id': None})

        for row in result_set:
            if len(url_dict[row.full_url]['chunks']) < 3:  # only keep the top 3 chunks for each URL
                url_dict[row.full_url]['chunks'].append((row.chunk, row.score))
                url_dict[row.full_url]['title'] = row.title
                url_dict[row.full_url]['url_id'] = row.url_id

        # Convert dictionary to list
        return [{'full_url': url, **info} for url, info in url_dict.items()]

    def load_snapshot(self, user_id: uuid4, url_id: uuid1) -> tuple[uuid1, str, str, str]:
        query = self.session.prepare(
            f"""
            SELECT url_id, url, title FROM {self.keyspace}.{self.table_urls} 
            WHERE user_id = ? AND url_id = ?
            """
        )
        # sort the results because we can't do it in CQL
        rows = self.session.execute(query, (user_id, url_id)).all()
        rows.sort(key=lambda row: row.url_id, reverse=True)
        url_id, url, title = rows[0]
        parsed = urlparse(url)
        path = parsed.hostname + parsed.path

        query = self.session.prepare(
            f"""
            SELECT text_content, formatted_content
            FROM {self.keyspace}.{self.table_paths} 
            WHERE user_id = ? AND url_id = ? AND path = ?
            """
        )
        row = self.session.execute(query, (user_id, url_id, path)).one()
        return url_id, title, row.text_content, row.formatted_content

    def save_formatting(self, user_id: uuid4, url_id: uuid1, path: str, formatted_content: str) -> None:
        request = self.session.prepare(
            f"""
            UPDATE {self.keyspace}.{self.table_paths}
            SET formatted_content = ?
            WHERE user_id = ? AND url_id = ? AND path = ?
            """
        )
        self.session.execute(request, (formatted_content, user_id, url_id, path))

    def _get_user_ids(self):
        return  self.session.execute(f"SELECT user_id FROM {self.keyspace}.{self.table_chunks}").all()
