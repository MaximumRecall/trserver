import os

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

from .db import DB


# configure for astra or localhost
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