# Rooter Experiments

Experiment orchestration code lives here.

This repo does not own dataset files and does not change production interfaces.
It reads the dataset manifest, calls existing tools through configured commands,
and logs each experiment run to MLflow.

## Artifact Storage

MLflow is already running at:

```text
http://127.0.0.1:5000
```

The current server stores artifacts under:

```text
/home/ec2-user/mlflow_server/mlartifacts
```

The runner also writes a local scratch copy under `outputs/`, which is ignored by
git. MLflow is the durable experiment store.

## Setup

```bash
cd /home/ec2-user/rooter-experiments
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Postprocessor Experiment

Copy the example config and set the real postprocessor command:

```bash
cp configs/postprocessor.example.json configs/postprocessor.local.json
```

Then run:

```bash
python runners/run_postprocessor.py --config configs/postprocessor.local.json
```

Preview selected app runs and rendered commands without executing anything:

```bash
python runners/run_postprocessor.py --config configs/postprocessor.local.json --dry-run
```

The command template can use these placeholders:

```text
{raw_json_path}
{frames_path}
{output_dir}
{session_id}
{app_run_id}
```

Each selected app run becomes one MLflow run with dataset/postprocessor git
commits, app model family tags, config params, and output artifacts.
