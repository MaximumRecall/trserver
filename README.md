This is the server for Total Recall.

It can run against a local vector-search-enabled Cassandra, or against DataStax Astra.

To run locally, 

`$ flask --app flaskr.app run --debug --host=0.0.0.0`

Environment variables that must be set:
`OPENAI_KEY`

The service will connect to local Cassandra by default (see config.py); to connect to Astra instead, 
specify the following environment variables:
`ASTRA_CLIENT_ID`
`ASTRA_CLIENT_SECRET`
