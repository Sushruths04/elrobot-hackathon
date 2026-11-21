# A100 policy server

Serves a LeRobot policy (default ACT) over HTTP. Schema: `src/normakit/policy_api.py` (state 8 floats 0..1, `image_jpeg_b64` table camera, returns `action_chunk` N x 8, `chunk_dt`, `server_ms`).

Endpoints: `GET /health`, `POST /act`, `POST /reset`, `GET /metrics`.
Auth: set `POLICY_API_TOKEN` on the server and the client; requests need `Authorization: Bearer <token>`. Unset = open (use an SSH tunnel or firewall).

## Docker (A100 host)
```
git clone <repo> && cd normacore-hackathon
docker build -f server/Dockerfile -t normakit-policy .
docker run --gpus all -p 8000:8000 -e CHECKPOINT=<hf_user>/<act_checkpoint> \
  -e POLICY_TYPE=act -e POLICY_API_TOKEN=<token> -v $HOME/models:/models normakit-policy
```
Local checkpoint dir: add `-v /path/ckpt:/ckpt -e CHECKPOINT=/ckpt`. Other types: `POLICY_TYPE=smolvla|diffusion|pi0`.

## Without Docker
```
CHECKPOINT=<hf_user>/<act_checkpoint> POLICY_TYPE=act PORT=8000 DEVICE=cuda server/run.sh
```

## Dummy mode (no GPU, for wiring tests)
```
.venv/bin/python server/policy_server.py --dummy --port 8765
```

## Benchmark
```
POLICY_API_TOKEN=<token> python server/bench.py --url http://A100_HOST:8000 --n 100
```
Reach the A100 via `ssh -L 8000:localhost:8000 <user>@A100_HOST` and use `--url http://localhost:8000`.

Note: the LeRobot loading path (`from_pretrained`, pre/post processors) is untested on a GPU; verify on the A100 first.
