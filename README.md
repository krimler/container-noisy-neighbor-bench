# container-noisy-neighbor-bench

A testbed for measuring when (and whether) noisy neighbor effects are actually detectable in Docker containers.

## What this is

Five Docker containers sharing 2 CPUs, with configurable background load, fault injection, and metric collection. You inject CPU/memory/IO load into one container and measure what happens to the others. The included parameter sweep tests different isolation modes and concurrency levels to find out which signals are reliable.

## Why

The "noisy neighbor" problem is well-discussed in container orchestration, but it is rarely shown empirically what is and isn't detectable from the outside. Most monitoring guides say "watch your CPU and memory metrics." This testbed shows when that advice works and when it doesn't.

## Key findings

We ran 36 trials across 3 isolation modes, 3 concurrency levels, and 2 different noisy neighbors. Here is what we found.

### CPU metrics are unreliable at normal utilization

At low concurrency (4 threads per container), `docker stats` CPU% correctly identifies the noisy neighbor 100% of the time. At 8 or 16 threads per container, it drops to 17%. The Linux CFS scheduler gives each container its proportional CPU share regardless of how many threads are fighting inside that share. The noisy neighbor's CPU% looks the same as everyone else's.

```
Concurrency    CPU delta detects NN    Latency detects NN
4 threads      100%                    92%
8 threads      17%                     92%
16 threads     17%                     83%
```

### Application latency is the most consistent signal

The noisy neighbor's own application latency spikes +90% to +220% across all configurations. Its internal burn threads compete with the application threads for the same CFS allocation, so response times increase even though Docker CPU% stays flat.

```
Isolation mode    CPU detection    Latency detection    Mean NN lat increase
cpu_shares        67%              75%                  +90%
equal_shares      33%              92%                  +162%
no_limits         33%              100%                 +220%
```

### Throughput collapse is the cleanest indicator

In every single trial, the noisy neighbor's throughput dropped from ~28-30 req/s to ~9-11 req/s (about 60-65% reduction). No other container showed this pattern. We weren't originally measuring this as a detection signal, but it turned out to be the most consistent one in the data.

### Cross-container effects exist but are noisy

About 78% of trials showed at least one other container with latency increases above 15%. But which containers were affected varied between trials, making this a supporting signal rather than a primary one.

## Architecture

```
+--------------+  +--------------+  +--------------+
| api-gateway  |  | user-service |  |order-service |
|   :5001      |  |   :5002      |  |   :5003      |
+--------------+  +--------------+  +--------------+
+--------------+  +--------------+
|payment-svc   |  |batch-process |
|   :5004      |  |   :5005      |
+--------------+  +--------------+
        |               |
   Background load (configurable threads per container)
   Metrics collector (Docker stats + HTTP latency probes)
```

Each container runs a Flask app with:
- `/work` endpoint: 50k MD5 hash operations (simulates CPU-bound work)
- `/health` endpoint: basic health check
- `/inject` endpoint: starts/stops CPU, memory, or IO stress threads

## Setup

Requires Docker Desktop with 2 CPUs allocated (Settings > Resources).

```bash
pip install -r requirements.txt
docker compose up -d --build
```

## Running

### Quick demo (3 trials)

```bash
python main.py demo --no-llm
```

Starts background load, picks a random noisy neighbor, injects CPU stress, and compares threshold-based detection (System A) against cross-container delta analysis (System C).

### Full batch experiment (30 trials)

```bash
python batch_experiment.py
```

Runs 6 trials per container, outputs `batch_results.csv` and breakdown by container and load type.

### Parameter sweep (36 trials, ~25 min)

```bash
python sweep.py
```

Tests 3 isolation modes (cpu_shares with unequal weights, equal shares, no limits) at 3 concurrency levels (4, 8, 16 threads per container) with 2 different noisy neighbors. Switches Docker Compose configs automatically. Outputs `sweep_results.csv`.

### Tuning

You can adjust the background load:

```bash
python main.py --concurrency 4 --think-time 0.1 demo --no-llm
```

Lower concurrency = more idle CPU = easier to detect. Higher concurrency = more realistic contention = harder to detect. That tradeoff is the whole point.

## Isolation modes

Three Docker Compose files test different cgroup configurations:

- `docker-compose.shares.yml`: Unequal cpu_shares. batch-processor gets 1024, others get 256. This is the only mode where CPU delta detection works consistently, because the high-share container visibly gains CPU when stressed.
- `docker-compose.equalshares.yml`: All containers get equal cpu_shares (512). CPU detection drops to 33% because no container can gain more CPU than its equal share.
- `docker-compose.nolimits.yml`: No CPU constraints at all. Similar to equal shares in practice because CFS distributes fairly by default.

## Detection approaches

**System A (threshold monitoring):** Alerts when any container exceeds CPU or latency thresholds. Detects that something is wrong 100% of the time but blames the victim (the container showing high latency) rather than the cause. This is how most basic monitoring works.

**System C (cross-container delta analysis):** Compares each container's metrics during contention against its own baseline. Looks for the container whose CPU increased the most relative to its own normal state. Works well for batch-processor (100% correct across 6 trials) but fails for equal-share containers (0% correct) because CFS hides the CPU increase.

## File overview

```
sweep.py                        # Parameter sweep across configs
batch_experiment.py              # 30-trial structured experiment
main.py                         # CLI for quick demos
agent/
  analyzer.py                   # Detection algorithms (System A + C)
  collector.py                  # Docker stats + latency probe collection
  experiment.py                 # Trial runner and evaluation
  loadgen.py                    # Background load generator
  rag.py                        # ChromaDB knowledge base for config context
services/
  app.py                        # Flask app (work, health, inject endpoints)
  Dockerfile
knowledge/
  container_configs.json         # Container config docs for RAG queries
docker-compose.shares.yml        # Unequal cpu_shares
docker-compose.equalshares.yml   # Equal cpu_shares
docker-compose.nolimits.yml      # No CPU constraints
```

## Optional: LLM diagnosis

If you have Ollama running with llama3.2, drop the `--no-llm` flag and System C will generate a natural-language diagnosis grounded in the RAG knowledge base. This is experimental.

```bash
ollama pull llama3.2
python main.py demo
```

## What you might use this for

- Testing your own monitoring setup against a known ground truth
- Understanding how CFS scheduling affects container observability
- Evaluating whether a detection approach works at different utilization levels
- Teaching or demos about container resource isolation

## Limitations

- Docker Desktop on Mac runs containers in a Linux VM, which adds a layer of abstraction. Results on bare-metal Linux may differ.
- The 2-CPU constraint forces contention but is not representative of production hosts.
- Workloads are synthetic (hash operations). Real applications with network calls and database queries would behave differently.
- The metric collector polls at 1-second intervals. Sub-second effects are not captured.
