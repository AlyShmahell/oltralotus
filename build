#!/bin/bash 
# echo '{"runtimes":{"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}},"default-runtime":"nvidia"}' | sudo tee /etc/docker/daemon.json > /dev/null && sudo systemctl restart docker
sudo docker system prune -f 
sudo -E  docker compose down --remove-orphans
sudo -E  docker compose up -d --build --force-recreate
sudo docker compose logs -f