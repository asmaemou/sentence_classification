# transformer_baseline.py
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast

# 1. Load the dataset
data_path = "../data/Amazon_reviews/train_cleaned.csv"
df = pd.read_csv(data_path)
print("Available columns:", df.columns.tolist())

# Use the correct column for the review text.
# Checking in the following order: 'cleaned_review', 'clean_review', 'review_text', 'review', 'text'
if 'cleaned_review' in df.columns:
    texts = df['cleaned_review'].astype(str).tolist()
elif 'clean_review' in df.columns:
    texts = df['clean_review'].astype(str).tolist()
elif 'review_text' in df.columns:
    texts = df['review_text'].astype(str).tolist()
elif 'review' in df.columns:
    texts = df['review'].astype(str).tolist()
elif 'text' in df.columns:
    texts = df['text'].astype(str).tolist()
else:
    raise ValueError("No valid text column found. Expected one of: 'cleaned_review', 'clean_review', 'review_text', 'review', or 'text'.")

# Use the correct column for labels.
# Checking 'label' first, then 'sentiment'
if 'label' in df.columns:
    labels = df['label'].tolist()
elif 'sentiment' in df.columns:
    labels = df['sentiment'].tolist()
    # Try converting labels to integers.
    try:
        # If they are numeric strings or integers, this will work.
        labels = [int(x) for x in labels]
    except Exception as e:
        # Otherwise, map common string values to integers.
        label_map = {"positive": 1, "negative": 0}
        try:
            labels = [label_map[x.strip().lower()] for x in labels]
        except Exception as ex:
            raise ValueError("Could not convert sentiment labels to integers. Please ensure they are either integers or 'positive'/'negative'.")
else:
    raise ValueError("No valid label column found. Expected one of: 'label' or 'sentiment'.")

# For debugging / quick runs, you might limit the data:
DEBUG = False
if DEBUG:
    texts = texts[:2000]
    labels = labels[:2000]

# 2. Train/Validation Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# 3. Load a Pre-trained Transformer and its Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Tokenize the Data
def tokenize(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

# 5. Create tf.data.Dataset Objects
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))

batch_size = 16
train_dataset = train_dataset.shuffle(1000).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# 6. Compile the Model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# 7. Train the Model
num_epochs = 3  # Fine-tuning often converges in a few epochs
history = model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)

# 8. Plot Training Metrics
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Transformer Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Transformer Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("transformer_training_metrics.png")
plt.close()
print("Saved transformer training metrics plot to transformer_training_metrics.png")

# 9. Generate a Confusion Matrix
val_preds_logits = model.predict(val_dataset).logits
val_preds = np.argmax(val_preds_logits, axis=1)

cm = confusion_matrix(val_labels, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Transformer Confusion Matrix")
plt.savefig("transformer_confusion_matrix.png")
plt.close()
print("Saved transformer confusion matrix plot to transformer_confusion_matrix.png")
