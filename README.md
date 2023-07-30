# openmetrics-cli-reader
This is a command line utility that scrapes an endpoint in openmetrics text format and output the metrics as JSON. I mostly use this to pipe the output into other things for processing, like nushell or jq.
It accepts a list of one or more URLs to read metrics from, and optionally accepts an interval in seconds when it should poll those endpoints. For example to read a single URL once:
```
$ read-metrics http://localhost:8000/metrics
```
Or to read from several separate URLs, and poll them every 10 seconds:
```
$ read-metrics --interval 10 http://localhost:8000/metrics http://localhost:8001/metrics http://localhost:8002/metrics
```
When reading from multiple URLs, metrics can be differentiated through the additional label "metrics_source", which is set to the respective URL.
