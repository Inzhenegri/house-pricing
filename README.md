# A regression example

The example shows the prediction of California houses' price.

This example was coded in order to present it on the WorldSkills Asia 2021 competition. I wrote this by my friends' demand. However, right now this project stands for a simple example of a regression and doesn't belong to any competition. All the code I wrote by myself, investigating and searching about different things like loss functions, the amount of neurons in each layer, activation functions, math equations and so on.

## Run in a Docker container
To run the main file in a Docker container, use this command:
```bash
docker run -u $(id -u):$(id -g) -it --rm -v $(pwd):/home/house_pricing/ arsenydeveloper/custom-tensorflow:2.4.2-gpu python3 /home/house_pricing/main.py
```

## Run with no Docker
```python
python3 main.py
```
