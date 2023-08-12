from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the model
model = load_model('predict_binary_model.h5')

# Plot the model
plot_model(model, to_file='binary_classification_model.png', show_shapes=True)
