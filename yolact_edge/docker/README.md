## Use the container

```Shell
cd docker/

# Build:
docker build --build-arg USER_ID=$UID -t yolact_edge_image .

# Launch (with GPUs):
./start.sh /path/to/yolact_edge /path/to/datasets
```
