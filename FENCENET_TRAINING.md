# Specification: Training Pipeline for FenceNet

## 1. Objective
Develop the training pipeline for the `FenceNet` action recognition model using PyTorch. This includes implementing the custom Dataset for 2D pose data, defining the Temporal Convolutional Network (TCN) architecture, configuring the training loop, and setting up the evaluation metrics. Ensure the code includes robust device mapping (e.g., `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) for seamless transitioning between local CPU prototyping and GPU server execution.

## 2. Dataset and Dataloader (`src/data/fencing_dataset.py`)
* **Input Data Structure:** The raw data consists of extracted 2D skeletal keypoints for fencing videos.
* **Target Classes:** 6 classes (`0: R`, `1: IS`, `2: WW`, `3: JS`, `4: SF`, `5: SB`).
* **Feature Extraction:**
    * Extract exactly 9 joints: front wrist, front elbow, front shoulder, both hips, both knees, both ankles.
    * Each joint has $x$ and $y$ coordinates. Total channels = 18.
* **Spatial Normalization (Crucial):**
    * For every frame, subtract the nose coordinates of the *first frame* ($t=0$) from all joint coordinates.
    * Divide the result by the vertical distance between the head and the front ankle in the *first frame*.
* **Temporal Sampling:**
    * During training, dynamically sample a subsequence of exactly 28 consecutive frames.
    * The starting frame of the sample should be randomly selected between frame 0 and frame 20.
* **Output Tensor Shape:** The `__getitem__` method must return a tuple `(features, label)` where `features` is a `torch.FloatTensor` of shape `(18, 28)`.

## 3. Model Architecture (`src/models/fencenet.py`)
* **Base Block (`TCNBlock`):**
    * Must use 1D Convolutions (`nn.Conv1d`).
    * Must enforce **Causality**: Pad the left side of the input sequence with `(kernel_size - 1) * dilation` zeros, and truncate the right side after convolution to ensure no future information leaks.
    * Structure per block: 
        1. Dilated Causal Conv1D
        2. Weight Normalization (`torch.nn.utils.weight_norm`)
        3. ReLU
        4. Spatial Dropout (p=0.2)
        5. Repeat 1-4 for a second convolutional layer.
        6. Residual connection: `output = relu(input + block_output)`. (Use a 1x1 Conv1D if input and output channels differ).
* **FenceNet Structure:**
    * Stack 6 `TCNBlock`s. (Note: Adjust channel sizes progressively, e.g., 18 -> 32 -> 64 -> 128, while increasing dilation rates exponentially, e.g., 1, 2, 4, 8...).
    * From the output of the final TCN block, extract only the **last time step** `out[:, :, -1]`.
    * Pass this through a Dense Layer (Linear) with 64 units and ReLU activation.
    * Pass through a final Dense Layer with 6 units (logits) for classification.

## 4. Training Configuration (`train_fencenet.py`)
* **Loss Function:** `nn.CrossEntropyLoss()`.
* **Optimizer:** `torch.optim.Adam`.
* **Hyperparameters:**
    * Batch Size: 32 or 64.
    * Learning Rate: Start with 1e-3, implement a learning rate scheduler (e.g., `ReduceLROnPlateau`).
    * Epochs: Target ~100 epochs, but include early stopping.
* **Evaluation Strategy (10-Fold Cross-Validation):**
    * Implement logic for Leave-One-Person-Out cross-validation (assuming the dataset has fencer IDs). The model must be trained on 9 fencers and evaluated on the 1 unseen fencer to test generalization.
* **Inference/Validation Logic:**
    * During validation, a single video might be split into multiple 28-frame subsequences.
    * The model should predict the class for each subsequence.
    * The final predicted class for the video is determined by **Majority Voting** across all subsequences of that video.

## 5. Artifacts and Logging
* Save the best model weights to `weights/fencenet/best_model.pth` based on validation accuracy.
* Log training loss, validation loss, and a confusion matrix per epoch to monitor if the model struggles with distinguishing similar classes like `IS` (Incremental Speed) and `WW` (With Waiting).