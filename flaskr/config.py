import os
from pathlib import Path

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

from .db import DB


# Load secret keys
_secrets_dir = Path('secrets')
if not _secrets_dir.is_dir():
    raise(Exception('Secrets directory not found'))
for secret_file in _secrets_dir.iterdir():
    if secret_file.is_file():
        env_var_name = secret_file.name.upper()
        with open(secret_file, 'r') as f:
            secret_value = f.read().strip()
        os.environ[env_var_name] = secret_value


# Configure DB for astra or localhost
astra_client_id = os.environ.get('ASTRA_CLIENT_ID')
if astra_client_id:
    print('Connecting to Astra')
    cwd = os.path.dirname(os.path.realpath(__file__))
    cloud_config = {
      'secure_connect_bundle': os.path.join(cwd, 'secure-connect-total-recall.zip')
    }
    astra_client_secret = os.environ.get('ASTRA_CLIENT_SECRET')
    if not astra_client_secret:
        raise Exception('ASTRA_CLIENT_SECRET environment variable not set')
    auth_provider = PlainTextAuthProvider(astra_client_id, astra_client_secret)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    db = DB(cluster)
    tr_data_dir = '/home/ubuntu/trserver/data'
else:
    print('Connecting to local Cassandra')
    db = DB(Cluster())
    tr_data_dir = '/home/jonathan/Projects/trserver/data'