import pandas as pd
import numpy as np
import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load and clean dataset
df = pd.read_csv("phishing.csv")
df.dropna(inplace=True)

# Preprocess URLs
def clean_url(url):
    url = re.sub(r"https?://", "", url)  # remove http or https
    url = url.lower().strip()
    return url

df['url'] = df['url'].apply(clean_url)

X = df['url']
y = df['is_phishing'].astype(int)

# Tokenization
vocab_size = 15000
max_len = 150

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

# Save tokenizer
os.makedirs("lstm_model", exist_ok=True)
with open("lstm_model/tokenizer.json", "w") as f:
    f.write(json.dumps(tokenizer.to_json()))

X_seq = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_seq, maxlen=max_len, padding='post')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, stratify=y, random_state=42)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Build model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_ckpt = ModelCheckpoint("lstm_model/lstm_model.h5", save_best_only=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, model_ckpt],
    verbose=1
)

# Predict & evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)

report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'])
print(report)

# Save report
with open("lstm_model/classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.title("Confusion Matrix")
plt.savefig("lstm_model/confusion_matrix.png")
plt.close()

print("✅ Training complete. Results saved in 'lstm_model/'")
