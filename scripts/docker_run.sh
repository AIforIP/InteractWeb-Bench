docker run -it --rm \
  --name interactweb_test \
  --network host \
  -v ./src:/app/src \
  -v ./data:/app/data \
  -v ./experiment_results:/app/experiment_results \
  -v .env:/app/.env \
  -v config.yaml:/app/config.yaml \
  --entrypoint /bin/bash \
  interactweb-bench:v1.0
