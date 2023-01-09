import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

print(tf.__version__)
print("--------------------------------")

### Download
if 0:
    # Run this block only once
    url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
    dataset = tf.keras.utils.get_file("data", url,
                                        untar=True, cache_dir='./data',
                                        cache_subdir='')
dataset = './data'
dataset_dir = os.path.join(os.path.dirname(dataset), 'data')
print(os.listdir(os.path.dirname(dataset)))
print(os.listdir(dataset_dir))

print("--------------------------------")

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

sample_file = os.path.join(train_dir, 'csharp/0.txt')
with open(sample_file) as f:
  print(f.read())

# Dense = length of (javascript, python, csharp, java)
NUMBER_OF_LABELS = 4

print("--------------------------------")


### Load the dataset
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    f'{dataset}/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)



print(type(raw_train_ds))

datasetdummy = tf.data.Dataset.from_tensor_slices([1, 2, 3])
print(type(datasetdummy))

datasetdummy = datasetdummy.batch(3)
print(type(datasetdummy))


for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])






print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])







raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    f'{dataset}/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)


raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    f'{dataset}/test',
    batch_size=batch_size)



#Prepare the dataset for training
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)




# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)



def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label





# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))


print("1287 ---> ",vectorize_layer.get_vocabulary()[107])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[1139])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))







train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


#Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model
embedding_dim = 26

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(NUMBER_OF_LABELS)])

model.summary()


# Loss function and optimizer
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

#Train the model
epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


# Evaluate the model
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)




# Create a plot of accuracy and loss over time
history_dict = history.history
history_dict.keys()



acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()




plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()



## Export the model
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)







# Inference on new data

examples = [
    "javascript clause  typescript, coffee script and v8 engine.",
    "Eclipse Jsoup JMS Jboss DBCP and C3P0 Javassist and CgLib springboot requires? Java18? JDE? JDK? Log4j2 HttpClient JAXB Apache POI",
    "csharp constructor   microsoft vba, .net and mecab csharp.",
    "python has a function as def.  __init__.py and setup.py  pypi modules. keras and tensorflow. BeautifulSoup pymysql"
]

res = export_model.predict(examples)

for i, items in enumerate(res):
    max_value = max(items)
    index = list(items).index(max_value)
    label = raw_train_ds.class_names[index]
    print(f"{i}: {label} has {max_value} %: {examples[i]}")


