---
name: Bug report
about: The model kit is not behaving as expected, is producing an error, or a check does not match the contract.
title: ""
labels: "bug"
assignees: ""
---

## Environment

- Hardware: <!-- e.g. 1x DGX Spark (GB10, 128 GB unified memory) -->
- Memory available before launch: <!-- MemAvailable from /proc/meminfo -->
- Image: <!-- `docker images | grep tensorfold-nemotron-bolt` -->
- `start.sh` invocation: <!-- e.g. `./start.sh`, or `SERVE=1 ./start.sh` -->
- Changed settings: <!-- CONTEXT, PARALLEL, KV_DTYPE, SERVE, MODEL_REVISION -->

## Steps to Reproduce

1. Run `./start.sh`

## What happened

<!-- The script's output, and `docker logs nemotron35-bolt-tf` when SERVE=1. -->
