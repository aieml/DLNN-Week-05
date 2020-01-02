# DLNN-Week-05
In this week trained a CNN to identify cat &amp; dogs and tested it with some unseen data. We experienced that the CNN is suffering from over-fitting while training and ended up with a low validation accuracy like 75%. In coming weeks we will discuss about regularization methods for minimizing and avoiding over-fitting such dropout, early stopping, batch normalization and etc. As the 2nd In class project we implemented the NVIDIA self driving car model with Udacity Self Driving Car Simulator.

## Credits & Links

1. Siraj Raval's Youtube Toturial on [Self Driving Car](https://www.youtube.com/watch?v=EaY5QiZwSP4&t=556s)
2. Siraj Raval's Github repository on [Self Driving Car](https://github.com/llSourcell/How_to_simulate_a_self_driving_car)
3. Naoki Shibuya's  Github repository on [Self Driving Car](https://github.com/naokishibuya/car-behavioral-cloning)
4. Image Augmentation [tutorial](https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec)
5. End to End Learning for Self-Driving Cars [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by NVIDIA

## Modifications

1. Edit the ```load_data``` function in ```model.py``` in ```car-behavioral-cloning-master``` as mentioned below before training with your own data.

```python
def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv')).values

    X = data_df[:,:3]
    y = data_df[:,3]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

```
2. replace the ```mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1``` line in ```utils.py``` with the following.
```python
def random_shadow(image):
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1
```

## Commands (in Anaconda Prompt)

1. Activating the environment ```conda activate car-behavioral-cloning```
2. Training the model ```python model.py -d "path to where IMG folder and driving_log.csv are located"```
3. Runing the model ```python drive.py (name of the trained weight file)``` you need to open the Udacity Simulator in Autonomous mode first
