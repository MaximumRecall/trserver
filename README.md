This is the server for Total Recall.

It can run against a local vector-search-enabled Cassandra, or against DataStax Astra.

To run locally, 

`$ flask run --debug --host=0.0.0.0`

Environment variables that must be set:
`OPENAI_KEY`

If running against Astra, also:
`ASTRA_CLIENT_ID`
`ASTRA_CLIENT_SECRET`
