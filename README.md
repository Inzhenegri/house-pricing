# A regression example

## Run main.py
To run the main file in a Docker container, use this command:
```bash
docker run -u $(id -u):$(id -g) -it --rm -v $(pwd):/home/house_pricing/ arsenydeveloper/tensorflow:2.4.2-gpu python3 /home/house_pricing/main.py
```
