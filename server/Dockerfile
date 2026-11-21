# Build from the repo root: docker build -f server/Dockerfile -t normakit-policy .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY src/normakit/policy_api.py src/normakit/policy_api.py
COPY server/policy_server.py server/policy_server.py
# policy_server imports normakit.policy_api only; provide the package marker
RUN touch src/normakit/__init__.py
ENV HOST=0.0.0.0 PORT=8000 POLICY_TYPE=act DEVICE=auto HF_HOME=/models
EXPOSE 8000
CMD ["sh", "-c", "python server/policy_server.py --checkpoint \"$CHECKPOINT\" --policy-type \"$POLICY_TYPE\" --device \"$DEVICE\" --host \"$HOST\" --port \"$PORT\""]
