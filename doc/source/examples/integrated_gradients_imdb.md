# Integrated gradients for text classification on the IMDB dataset

In this example, we apply the integrated gradients method to a sentiment analysis model trained on the IMDB dataset. In text classification models, integrated gradients define an attribution value for each word in the input sentence. The attributions are calculated considering the integral of the model  gradients with respect to the word embedding layer along a straight path from a baseline instance $x^\prime$ to the input instance $x.$ A description of the method can be found [here](https://docs.seldon.io/projects/alibi/en/stable/methods/IntegratedGradients.html). Integrated gradients was originally proposed in Sundararajan et al., ["Axiomatic Attribution for Deep Networks"](https://arxiv.org/abs/1703.01365)

The IMDB data set contains 50K movie reviews labelled as positive or negative. 
We train a convolutional neural network classifier with a single 1-d convolutional layer followed by a fully connected layer. The reviews in the dataset are truncated at 100 words and each word is represented by 50-dimesional word embedding vector. We calculate attributions for the elements of the embedding layer.

<div class="alert alert-info">
Note
    
To enable support for IntegratedGradients, you may need to run
    
```bash
pip install alibi[tensorflow]
```

</div>


```python
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout 
from tensorflow.keras.utils import to_categorical
from alibi.explainers import IntegratedGradients
import matplotlib.pyplot as plt
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # True
```

    TF version:  2.6.0
    Eager execution enabled:  True


## Load data

Loading the imdb dataset. 


```python
max_features = 10000
maxlen = 100
```


```python
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
test_labels = y_test.copy()
train_labels = y_train.copy()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

index = imdb.get_word_index()
reverse_index = {value: key for (key, value) in index.items()} 
```

    Loading data...
    25000 train sequences
    25000 test sequences
    Pad sequences (samples x time)
    x_train shape: (25000, 100)
    x_test shape: (25000, 100)


A sample review from the test set. Note that unknown words are replaced with 'UNK'


```python
def decode_sentence(x, reverse_index):
    # the `-3` offset is due to the special tokens used by keras
    # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    return " ".join([reverse_index.get(i - 3, 'UNK') for i in x])
```


```python
print(decode_sentence(x_test[1], reverse_index)) 
```

    a powerful study of loneliness sexual UNK and desperation be patient UNK up the atmosphere and pay attention to the wonderfully written script br br i praise robert altman this is one of his many films that deals with unconventional fascinating subject matter this film is disturbing but it's sincere and it's sure to UNK a strong emotional response from the viewer if you want to see an unusual film some might even say bizarre this is worth the time br br unfortunately it's very difficult to find in video stores you may have to buy it off the internet


## Train Model

The model includes one convolutional layer and reaches a test accuracy of 0.85. If `save_model = True`, a local folder `../model_imdb` will be created and the trained model will be saved in that folder. If the model was previously saved, it can be loaded by setting `load_model = True`.


```python
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
```


```python
load_model = False
save_model = True
```


```python
filepath = './model_imdb/'  # change to directory where model is downloaded
if load_model:
    model = tf.keras.models.load_model(os.path.join(filepath, 'model.h5'))
else:
    print('Build model...')
    
    inputs = Input(shape=(maxlen,), dtype=tf.int32)
    embedded_sequences = Embedding(max_features,
                                   embedding_dims)(inputs)
    out = Conv1D(filters, 
                 kernel_size, 
                 padding='valid', 
                 activation='relu', 
                 strides=1)(embedded_sequences)
    out = Dropout(0.4)(out)
    out = GlobalMaxPooling1D()(out)
    out = Dense(hidden_dims, 
                activation='relu')(out)
    out = Dropout(0.4)(out)
    outputs = Dense(2, activation='softmax')(out)
        
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=256,
              epochs=3,
              validation_data=(x_test, y_test))
    if save_model:  
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        model.save(os.path.join(filepath, 'model.h5')) 
```

    Build model...
    Train...
    Epoch 1/3
    98/98 [==============================] - 13s 130ms/step - loss: 0.6030 - accuracy: 0.6534 - val_loss: 0.4228 - val_accuracy: 0.8192
    Epoch 2/3
    98/98 [==============================] - 14s 146ms/step - loss: 0.3223 - accuracy: 0.8631 - val_loss: 0.3489 - val_accuracy: 0.8542
    Epoch 3/3
    98/98 [==============================] - 19s 197ms/step - loss: 0.2128 - accuracy: 0.9177 - val_loss: 0.3327 - val_accuracy: 0.8545


## Calculate integrated gradients

The integrated gradients attributions are calculated with respect to the embedding layer for 10 samples from the test set. Since the model uses a word to vector embedding with vector dimensionality of 50 and sequence length of 100 words, the dimensionality of the attributions is (10, 100, 50). In order to obtain a single attribution value for each word, we sum all the attribution values for the 50 elements of each word's vector representation.
 
The default baseline is used in this example which is internally defined as a sequence of zeros. In this case, this corresponds to a sequence of padding characters (**NB:** in general the numerical value corresponding to a "non-informative" baseline such as the PAD token will depend on the tokenizer used, make sure that the numerical value of the baseline used corresponds to your desired token value to avoid surprises). The path integral is defined as a straight line from the baseline to the input image. The path is approximated by choosing 50 discrete steps according to the Gauss-Legendre method.


```python
layer = model.layers[1]
layer
```




    <keras.layers.embeddings.Embedding at 0x7fb80bdf1e50>




```python
n_steps = 50
method = "gausslegendre"
internal_batch_size = 100
nb_samples = 10
ig  = IntegratedGradients(model,
                          layer=layer,
                          n_steps=n_steps, 
                          method=method,
                          internal_batch_size=internal_batch_size)
```


```python
x_test_sample = x_test[:nb_samples]
predictions = model(x_test_sample).numpy().argmax(axis=1)
explanation = ig.explain(x_test_sample, 
                         baselines=None, 
                         target=predictions,
                         attribute_to_layer_inputs=False)
```


```python
# Metadata from the explanation object
explanation.meta
```




    {'name': 'IntegratedGradients',
     'type': ['whitebox'],
     'explanations': ['local'],
     'params': {'method': 'gausslegendre',
      'n_steps': 50,
      'internal_batch_size': 100,
      'layer': 1}}




```python
# Data fields from the explanation object
explanation.data.keys()
```




    dict_keys(['attributions', 'X', 'forward_kwargs', 'baselines', 'predictions', 'deltas', 'target'])




```python
# Get attributions values from the explanation object
attrs = explanation.attributions[0]
print('Attributions shape:', attrs.shape)
```

    Attributions shape: (10, 100, 50)


## Sum attributions


```python
attrs = attrs.sum(axis=2)
print('Attributions shape:', attrs.shape)
```

    Attributions shape: (10, 100)


## Visualize attributions


```python
i = 1
x_i = x_test_sample[i]
attrs_i = attrs[i]
pred = predictions[i]
pred_dict = {1: 'Positive review', 0: 'Negative review'}
```


```python
print('Predicted label =  {}: {}'.format(pred, pred_dict[pred]))
```

    Predicted label =  1: Positive review


We can visualize the attributions for the text instance by mapping the values of the attributions onto a matplotlib colormap. Below we define some utility functions for doing this.


```python
from IPython.display import HTML
def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"
```


```python
def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)
    
    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors
```

Below we visualize the attribution values (highlighted in the text) having the highest positive attributions. Words with high positive attribution are highlighted in shades of green and words with negative attribution in shades of pink. Stronger shading corresponds to higher attribution values. Positive attributions can be interpreted as increase in probability of the predicted class ("Positive sentiment") while negative attributions correspond to decrease in probability of the predicted class.


```python
words = decode_sentence(x_i, reverse_index).split()
colors = colorize(attrs_i)
```


```python
HTML("".join(list(map(hlstr, words, colors))))
```




<mark style=background-color:#f8f5f6>a </mark><mark style=background-color:#f1f6ea>powerful </mark><mark style=background-color:#f5f7f2>study </mark><mark style=background-color:#f4f7f0>of </mark><mark style=background-color:#f7f7f7>loneliness </mark><mark style=background-color:#f7f7f6>sexual </mark><mark style=background-color:#f7f7f6>UNK </mark><mark style=background-color:#f7f7f6>and </mark><mark style=background-color:#f7f7f6>desperation </mark><mark style=background-color:#f6f7f5>be </mark><mark style=background-color:#f4f7f0>patient </mark><mark style=background-color:#f7f6f7>UNK </mark><mark style=background-color:#f6f7f5>up </mark><mark style=background-color:#e2f3ca>the </mark><mark style=background-color:#529624>atmosphere </mark><mark style=background-color:#d6eeb6>and </mark><mark style=background-color:#faecf3>pay </mark><mark style=background-color:#f7f7f6>attention </mark><mark style=background-color:#f7f6f7>to </mark><mark style=background-color:#f5f7f3>the </mark><mark style=background-color:#2a681a>wonderfully </mark><mark style=background-color:#cbe9a4>written </mark><mark style=background-color:#99065a>script </mark><mark style=background-color:#fcdded>br </mark><mark style=background-color:#faeaf2>br </mark><mark style=background-color:#f8f4f6>i </mark><mark style=background-color:#f8f3f6>praise </mark><mark style=background-color:#f8f5f6>robert </mark><mark style=background-color:#fbe7f2>altman </mark><mark style=background-color:#f9eef4>this </mark><mark style=background-color:#edf6e1>is </mark><mark style=background-color:#f1f6ea>one </mark><mark style=background-color:#f3f7ef>of </mark><mark style=background-color:#f7f7f6>his </mark><mark style=background-color:#f6f7f5>many </mark><mark style=background-color:#f7f7f7>films </mark><mark style=background-color:#f6f7f5>that </mark><mark style=background-color:#f2f6ec>deals </mark><mark style=background-color:#f1f6e8>with </mark><mark style=background-color:#e1f3c7>unconventional </mark><mark style=background-color:#a1d26a>fascinating </mark><mark style=background-color:#f4f7f0>subject </mark><mark style=background-color:#f7f7f6>matter </mark><mark style=background-color:#f7f7f6>this </mark><mark style=background-color:#f9f1f5>film </mark><mark style=background-color:#f8f4f6>is </mark><mark style=background-color:#f8f4f6>disturbing </mark><mark style=background-color:#f7f7f6>but </mark><mark style=background-color:#f7f7f7>it's </mark><mark style=background-color:#f7f7f7>sincere </mark><mark style=background-color:#f6f7f5>and </mark><mark style=background-color:#f8f4f6>it's </mark><mark style=background-color:#f7f7f7>sure </mark><mark style=background-color:#f8f4f6>to </mark><mark style=background-color:#f5f7f3>UNK </mark><mark style=background-color:#eff6e5>a </mark><mark style=background-color:#498d20>strong </mark><mark style=background-color:#d6eeb6>emotional </mark><mark style=background-color:#f6f7f5>response </mark><mark style=background-color:#f6f7f5>from </mark><mark style=background-color:#f1f6e8>the </mark><mark style=background-color:#f1f6e8>viewer </mark><mark style=background-color:#f9f0f5>if </mark><mark style=background-color:#faecf3>you </mark><mark style=background-color:#fad4e9>want </mark><mark style=background-color:#f9eef4>to </mark><mark style=background-color:#f1f6ea>see </mark><mark style=background-color:#edf6e1>an </mark><mark style=background-color:#276419>unusual </mark><mark style=background-color:#e6f5d0>film </mark><mark style=background-color:#f0f6e7>some </mark><mark style=background-color:#fad6ea>might </mark><mark style=background-color:#ea9fca>even </mark><mark style=background-color:#fddeee>say </mark><mark style=background-color:#a40b63>bizarre </mark><mark style=background-color:#f3bdde>this </mark><mark style=background-color:#eef6e2>is </mark><mark style=background-color:#eff6e4>worth </mark><mark style=background-color:#f5f7f3>the </mark><mark style=background-color:#f7f7f7>time </mark><mark style=background-color:#faedf3>br </mark><mark style=background-color:#f7cbe4>br </mark><mark style=background-color:#dd73ac>unfortunately </mark><mark style=background-color:#f8f5f6>it's </mark><mark style=background-color:#edf6e1>very </mark><mark style=background-color:#f5f7f2>difficult </mark><mark style=background-color:#f7f7f6>to </mark><mark style=background-color:#f7f7f6>find </mark><mark style=background-color:#f7f7f6>in </mark><mark style=background-color:#f7f7f6>video </mark><mark style=background-color:#f7f7f7>stores </mark><mark style=background-color:#f3f7ef>you </mark><mark style=background-color:#f7f7f6>may </mark><mark style=background-color:#f7f7f7>have </mark><mark style=background-color:#f4f7f0>to </mark><mark style=background-color:#eaf5d9>buy </mark><mark style=background-color:#edf6e1>it </mark><mark style=background-color:#f3f6ed>off </mark><mark style=background-color:#f5f7f2>the </mark><mark style=background-color:#f4f7f0>internet </mark>


