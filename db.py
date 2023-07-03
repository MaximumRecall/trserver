import time
from collections import defaultdict
from datetime import datetime
from uuid import uuid4, uuid1
from typing import Any, Dict, List, Tuple, Union
from cassandra.cluster import Cluster, ResultSet
from cassandra.concurrent import execute_concurrent_with_args

class DB:
    def __init__(self, cluster: Cluster) -> None:
        self.keyspace = "total_recall"
        self.table_chunks = "saved_chunks"
        self.table_urls = "saved_urls"
        self.cluster = cluster
        self.session = self.cluster.connect()

        # Keyspace
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
            url text,
            title text,
            PRIMARY KEY (user_id, url_id));
            """
        )

        # Chunks table
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table_chunks} (
            user_id uuid,
            chunk_id timeuuid,
            url text,
            title text,
            chunk text,
            embedding vector<float, 384>,
            PRIMARY KEY (user_id, chunk_id));
            """
        )

        # Create SAI index if not exists
        sai_index_name = f"{self.table_chunks}_embedding_idx"
        self.session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {sai_index_name} ON {self.keyspace}.{self.table_chunks} (embedding)
            USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
            WITH OPTIONS = {{ 'similarity_function': 'dot_product' }}
            """
        )


    def upsert_chunks(self, user_id: uuid4, url: str, title: str, chunks: List[Tuple[uuid1, str, List[float]]]) -> None:
        st_urls = self.session.prepare(
            f"""
            INSERT INTO {self.keyspace}.{self.table_urls}
            (user_id, url_id, url, title)
            VALUES (?, ?, ?, ?)
            """
        )
        # TODO retry?
        url_uuid = uuid1()
        self.session.execute(st_urls, (user_id, url_uuid, url, title))

        st_chunks = self.session.prepare(
            f"""
            INSERT INTO {self.keyspace}.{self.table_chunks}
            (user_id, chunk_id, url, title, chunk, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """
        )
        denormalized_chunks = [(user_id, chunk_id, url, title, chunk, embedding)
                               for chunk_id, chunk, embedding in chunks]
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


    def recent_urls(self, user_id: uuid4) -> ResultSet:
        query = self.session.prepare(
            f"""
            SELECT url, title, toTimestamp(url_id) as saved_at 
            FROM {self.keyspace}.{self.table_urls} 
            WHERE user_id = ? LIMIT 10
            """
        )
        return self.session.execute(query, (user_id,))


    def search(self, user_id: uuid4, vector: List[float]) -> List[Dict[str, Union[str, datetime, List[str]]]]:
        limit = 10
        url_dict = defaultdict(lambda: {'chunks': [], 'title': None, 'saved_at': None})
        chunk_ids = []
        while len(url_dict) < limit:
            if chunk_ids:
                query = self.session.prepare(
                    f"""
                    SELECT url, title, chunk, chunk_id
                    FROM {self.keyspace}.{self.table_chunks} 
                    WHERE user_id = ? 
                    AND chunk_id NOT IN ?
                    ORDER BY embedding ANN OF ? LIMIT 10
                    """
                )
                result_set = self.session.execute(query, (user_id, chunk_ids, vector))
            else:
                query = self.session.prepare(
                    f"""
                    SELECT url, title, chunk, chunk_id
                    FROM {self.keyspace}.{self.table_chunks} 
                    WHERE user_id = ? 
                    ORDER BY embedding ANN OF ? LIMIT 10
                    """
                )
                result_set = self.session.execute(query, (user_id, vector))
            if not result_set:  # stop if no more results
                break
            for row in result_set:
                chunk_ids.append(row.chunk_id)  # keep track of the chunk_ids already processed
                if len(url_dict[row.url]['chunks']) < 3:  # only keep the top 3 chunks for each URL
                    url_dict[row.url]['chunks'].append(row.chunk)
                    if url_dict[row.url]['title'] is None:
                        url_dict[row.url]['title'] = row.title
                        url_dict[row.url]['saved_at'] = datetime.fromtimestamp(row.chunk_id.time)
        return [{'url': url, **info} for url, info in url_dict.items()]