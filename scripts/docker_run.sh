#!/bin/bash

# =================================================================
# 项目名称：InteractWeb-Bench 实验启动脚本
# 脚本功能：启动指定版本的 Docker 容器进行模型测试
# 镜像版本：interactweb-bench:v1.0
# =================================================================

# 定义宿主机项目根路径，便于后续路径映射维护
PROJECT_PATH="/data/shared/users/wangqiyao/InteractWeb-Bench"

docker run -it --rm \
  --name interactweb_test-qwen-retest \
  --network host \
  -v "${PROJECT_PATH}/src:/app/src" \
  -v "${PROJECT_PATH}/scripts:/app/scripts" \
  -v "${PROJECT_PATH}/data:/app/data" \
  -v "${PROJECT_PATH}/experiment_results:/app/experiment_results" \
  -v "${PROJECT_PATH}/.env:/app/.env" \
  -v "${PROJECT_PATH}/config.yaml:/app/config.yaml" \
  -v "${PROJECT_PATH}/config1.yaml:/app/config1.yaml" \
  -v "${PROJECT_PATH}/config2.yaml:/app/config2.yaml" \
  -v "${PROJECT_PATH}/config3.yaml:/app/config3.yaml" \
  -v "${PROJECT_PATH}/config4.yaml:/app/config4.yaml" \
  --entrypoint /bin/bash \
  interactweb-bench:v1.0