import time
from uuid import uuid4, uuid1
from typing import Any, Dict, List, Tuple
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args

class DB:
    def __init__(self, cluster: Cluster) -> None:
        self.keyspace = "total_recall"
        self.table = "saved_chunks"
        self.cluster = cluster
        self.session = self.cluster.connect()

        # Create keyspace if not exists
        self.session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH REPLICATION = {{ 'class': 'SimpleStrategy', 'replication_factor': 1 }}
            """
        )

        # Create table if not exists
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table} (
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
        sai_index_name = f"{self.table}_embedding_idx"
        self.session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {sai_index_name} ON {self.keyspace}.{self.table} (embedding)
            USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
            WITH OPTIONS = {{ 'similarity_function': 'dot_product' }}
            """
        )

    def upsert_batch(self, user_id: uuid4, url: str, title: str, chunks: List[Tuple[uuid1, str, List[float]]]) -> None:
        st = self.session.prepare(
            f"""
            INSERT INTO {self.keyspace}.{self.table}
            (user_id, chunk_id, url, title, chunk, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """
        )
        denormalized_chunks = [(user_id, chunk_id, url, title, chunk, embedding)
                               for chunk_id, chunk, embedding in chunks]
        backoff = 0.5
        while denormalized_chunks and backoff < 60:
            results = execute_concurrent_with_args(self.session, st, denormalized_chunks,
                                                   concurrency=16, raise_on_first_error=False)
            denormalized_chunks = [chunk for chunk, (success, _)
                                   in zip(denormalized_chunks, results) if not success]
            time.sleep(backoff)
            backoff *= 2
        if denormalized_chunks:
            raise Exception(f"Failed to insert {len(denormalized_chunks)} chunks")

    def query(self, user_id: uuid4, vector: List[float], top_k: int) -> List[str]:
        pass
        # query = SimpleStatement(
        #     f"SELECT id, start, end, text FROM {self.keyspace}.{self.table} ORDER BY embedding ANN OF %s LIMIT %s"
        # )
        # res = self.session.execute(query, (vector, top_k))
        # rows = [row for row in res]
        # # print('\n'.join(repr(row) for row in rows))
        # return [row.text for row in rows]
