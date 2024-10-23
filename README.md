# Steps involved

### Step 1. Loading and Visualizing Audio Data
#### Objective: 
Load audio data and visualize it using Librosa and Matplotlib to understand the signal in the time domain.
#### Steps:
1. Load audio files using librosa.load().
2. Visualize the waveform using librosa.display.waveshow() and Matplotlib to plot the audio signal.

### Step 2. Feature Extraction Using MFCC
#### Objective: 
Extract Mel-Frequency Cepstral Coefficients (MFCC) from the audio files. MFCCs capture the audio's time-frequency characteristics and are crucial features for audio classification.
#### Steps:
1. Extract MFCCs using librosa.feature.mfcc().
2. Scale the MFCCs by taking the mean across the time axis.

### Step 3. Feature Extraction for the Entire Dataset
#### Objective: 
Extract MFCC features from all the audio files in the dataset and create a structured dataset for training the model.
#### Steps:
1. Load metadata (e.g., class labels) and iterate through each audio file.
2. Extract MFCCs for each file and append it to a list, along with the corresponding class label.
3. Convert the list of extracted features into a Pandas DataFrame.

### Step 4. Data Preparation for Model Training
#### Objective: 
Prepare the extracted features for model training by encoding class labels and splitting the dataset into training and testing sets.
#### Steps:
1. Convert the features and labels into NumPy arrays.
2. Encode the class labels using LabelEncoder and convert them to a categorical format.
3. Split the dataset into training and testing sets (80-20 split).

### Step 5. Model Creation
#### Objective: Create a deep learning model using a Sequential model in TensorFlow/Keras for audio classification.
#### Steps:
1. Define a Sequential model with fully connected (Dense) layers and dropout for regularization.
2. Use ReLU activation for the hidden layers and softmax for the output layer.
3. Compile the model with the categorical crossentropy loss function and the Adam optimizer.

### Step 6. Model Training
#### Objective: Train the model on the training data and save the best-performing model using ModelCheckpoint.
#### Steps:
1. Train the model for a specified number of epochs with a batch size of 32.
2. Use ModelCheckpoint to save the best model during training.
3. Measure the total training time.

### Step 7. Model Evaluation
#### Objective: Evaluate the trained model on the test data to determine its accuracy.
#### Steps:
1. Evaluate the model on the test set using model.evaluate().
2. Print the test accuracy.

### Step 8. Testing New Audio Data
### Objective: Test the model on new unseen audio data and predict its class.
#### Steps:
1. Preprocess the new audio data by extracting MFCC features.
2. Use the trained model to predict the class of the new audio.
3. Inverse transform the predicted label to get the corresponding class name.

