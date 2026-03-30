# docker run -it --rm \
#   --name interactweb_test-2 \
#   --network host \
#   -v ./scripts:/app/scripts \
#   -v ./src:/app/src \
#   -v ./data:/app/data \
#   -v ./experiment_results:/app/experiment_results \
#   -v .env:/app/.env \
#   -v config.yaml:/app/config.yaml \
#   --entrypoint /bin/bash \
#   interactweb-bench:v1.0

PROJECT_PATH="/data/shared/users/wangqiyao/Interactweb-Bench"

docker run -it --rm \
  --name interactweb_test-2 \
  --network host \
  -v "${PROJECT_PATH}/src:/app/src" \
  -v "${PROJECT_PATH}/data:/app/data" \
  -v "${PROJECT_PATH}/experiment_results:/app/experiment_results" \
  -v "${PROJECT_PATH}/.env:/app/.env" \
  -v "${PROJECT_PATH}/config.yaml:/app/config.yaml" \
  --entrypoint /bin/bash \
  interactweb-bench:v1.0
