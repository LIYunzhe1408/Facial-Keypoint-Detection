# Facial-Keypoint-Detection

## Direct Coordinate Regression
In this part, we implemented a simple CNN architecture to directly regress the coordinates of facial keypoints. The model will take a grayscale image as input and output a vector of length 2K, where K is the number of keypoints (68 in our case). This approach treats keypoint detection as a direct regression problem.
1. Analyze how different hyperparameters affect model performance
   * Learning Rate (lr)
     * Too high (e.g., 0.01 or more): Causes unstable training, loss may diverge.
     * Too low (e.g., 1e-5 or less): Slow convergence, might get stuck in local minima.
     * Experiments: Start with 1e-3, then decay or tune based on validation loss.
   * Dropout Rate
     * Too low (< 0.1): Not enough regularization → overfit.
     * Too high (> 0.5): Too much information dropped → underfit.
     * Increasing by layer depth (0.1 → 0.4) is a good pattern.
   * Batch Size
     * Large batch sizes: More stable gradients, faster per-epoch, but may generalize worse.
     * Small batch sizes: Better generalization, more noise in updates.
     * Try 32 or 64, and tune based on memory and validation performance.
   * Loss Function
     * SmoothL1Loss is robust to outliers (good for keypoints).
     * MSELoss (sensitive to large errors).
   * Epochs
     * Too few epochs: Underfitting, network hasn't converged.
     * Too many epochs: Overfitting unless using early stopping or validation loss monitoring.
   * Optimizer
     * Adam: fast convergence, less tuning.
     * SGD + momentum: may generalize better, needs learning rate scheduling.

## Workflow
1. Get dataset, split them into train, val, and test.
2. Define models.
3. Determine hyperparameters.
4. Train and val while training. Save best checkpoint, train_loss, val_loss `list`.
5. Plot training and val loss.
6. Load saved best model checkpoint. Run metric.
7. Load saved best model checkpoint. Run prediction, then visualize.

## TODO
1. Refine the training process for ResNet50 backbone. Current one for ResNet34 works not well even training for 100 epochs.
   * Learning rate for classifier when backbone is frozen and for entire model.
   * Scheduler?
   * Epochs
2. Adapt it to DiNO
3. Train more epochs for UNet
   * Sigma
   * Learning rate
