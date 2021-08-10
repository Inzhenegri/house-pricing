# A regression example

This example shows regression usage with tensorflow. All code is taken from this [tutorial](https://medium.com/analytics-vidhya/house-price-prediction-regression-with-tensorflow-keras-4fc49fae7123).

## Run main.py
To run the main file, use this command:
```bash
docker run -u $(id -u):$(id -g) -it --rm -v $(pwd):/home/house_pricing/ arsenydeveloper/tensorflow:2.4.2-gpu python3 /home/house_pricing/main.py
```
