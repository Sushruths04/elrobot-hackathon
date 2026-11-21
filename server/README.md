# A100 policy server

Serves a LeRobot policy (default ACT) over HTTP. Schema: `src/normakit/policy_api.py` (state 8 floats 0..1, `image_jpeg_b64` table camera, returns `action_chunk` N x 8, `chunk_dt`, `server_ms`).

Endpoints: `GET /health`, `POST /act`, `POST /reset`, `GET /metrics`.
Auth: set `POLICY_API_TOKEN` on the server and the client; every endpoint, including `/health` and `/metrics`, needs `Authorization: Bearer <token>`. The server refuses to start without a token unless `--dummy` or `--no-auth` is passed.

Network: the server speaks plain HTTP, so the bearer token is cleartext on the wire. It binds `127.0.0.1` by default (`--host` / env `HOST`; the Docker image sets `HOST=0.0.0.0` inside the container). Do not expose port 8000 to the internet. Reach it through an SSH tunnel (below), a VPN such as WireGuard/Tailscale, or a TLS reverse proxy (for example Caddy) in front of it. When publishing the Docker port, bind it to localhost: `-p 127.0.0.1:8000:8000`.

Checkpoint checks at load: the camera key is read from `policy.config.input_features` (one camera supported; `--image-key` / env `IMAGE_KEY` asserts the expected name), and state and action must be 8-dim. Each chunk is rejected (HTTP 500 with a message) if any action is NaN or outside -0.1..1.1, which catches checkpoints trained on degrees or raw ticks instead of 0..1 norms. A vision policy called without `image_jpeg_b64`, or a NaN state, gets HTTP 422.

## Docker (A100 host)
```
git clone <repo> && cd normacore-hackathon
docker build -f server/Dockerfile -t normakit-policy .
docker run --gpus all -p 127.0.0.1:8000:8000 -e CHECKPOINT=<hf_user>/<act_checkpoint> \
  -e POLICY_TYPE=act -e POLICY_API_TOKEN=<token> -v $HOME/models:/models normakit-policy
```
Local checkpoint dir: add `-v /path/ckpt:/ckpt -e CHECKPOINT=/ckpt`. Other types: `POLICY_TYPE=smolvla|diffusion|pi0`.

## Without Docker
```
POLICY_API_TOKEN=<token> CHECKPOINT=<hf_user>/<act_checkpoint> POLICY_TYPE=act PORT=8000 DEVICE=cuda server/run.sh
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
