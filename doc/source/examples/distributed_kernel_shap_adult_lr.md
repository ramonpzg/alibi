# Distributed KernelSHAP

<div class="alert alert-info">
Note

To enable SHAP support, you may need to run
    
```bash
pip install alibi[shap]
```

</div>

## Introduction

In this example, KernelSHAP is used to explain a batch of instances on multiple cores. To run this example, please run ``pip install alibi[ray]`` first.

<div class="alert alert-warning">
Warning

Windows support for the `ray` Python library is [in beta](https://docs.ray.io/en/latest/ray-overview/installation.html#windows-support). Using `KernelShap` in parallel is not currently supported on Windows platforms.


</div>


```python
# shap.summary_plot currently doesn't work with matplotlib>=3.6.0,
# see bug report: https://github.com/slundberg/shap/issues/2687
!pip install matplotlib==3.5.3
```


```python
import pprint
import shap
import ray
shap.initjs()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alibi.explainers import KernelShap
from alibi.datasets import fetch_adult
from collections import defaultdict
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from timeit import default_timer as timer
from typing import Dict, List, Tuple
```

## Data preparation

### Load and split

The `fetch_adult` function returns a `Bunch` object containing the features, the targets, the feature names and a mapping of categorical variables to numbers.


```python
adult = fetch_adult()
adult.keys()
```




    dict_keys(['data', 'target', 'feature_names', 'target_names', 'category_map'])




```python
data = adult.data
target = adult.target
target_names = adult.target_names
feature_names = adult.feature_names
category_map = adult.category_map
```

Note that for your own datasets you can use our utility function `gen_category_map` to create the category map.


```python
from alibi.utils import gen_category_map
```


```python
np.random.seed(0)
data_perm = np.random.permutation(np.c_[data, target])
data = data_perm[:,:-1]
target = data_perm[:,-1]
```


```python
idx = 30000
X_train,y_train = data[:idx,:], target[:idx]
X_test, y_test = data[idx+1:,:], target[idx+1:]
```

### Create feature transformation pipeline
Create feature pre-processor. Needs to have 'fit' and 'transform' methods. Different types of pre-processing can be applied to all or part of the features. In the example below we will standardize ordinal features and apply one-hot-encoding to categorical features.

Ordinal features:


```python
ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])
```

Categorical features:


```python
categorical_features = list(category_map.keys())
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('onehot', OneHotEncoder(drop='first', handle_unknown='error'))])
```

Note that in order to be able to interpret the coefficients corresponding to the categorical features, the option `drop='first'` has been passed to the `OneHotEncoder`. This means that for a categorical variable with `n` levels, the length of the code will be `n-1`. This is necessary in order to avoid introducing feature multicolinearity, which would skew the interpretation of the results. For more information about the issue about multicolinearity in the context of linear modelling see [[1]](#References).
<a id='src_1'></a>

Combine and fit:


```python
preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                               ('cat', categorical_transformer, categorical_features)])
preprocessor.fit(X_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 [0, 8, 9, 10]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;onehot&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;))]),
                                 [1, 2, 3, 4, 5, 6, 7, 11])])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 [0, 8, 9, 10]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;onehot&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;))]),
                                 [1, 2, 3, 4, 5, 6, 7, 11])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">num</label><div class="sk-toggleable__content"><pre>[0, 8, 9, 10]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">cat</label><div class="sk-toggleable__content"><pre>[1, 2, 3, 4, 5, 6, 7, 11]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop=&#x27;first&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div>



### Preprocess the data


```python
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
```

Applying the `sklearn` processing pipeline modifies the column order of the original dataset. The new feature ordering is necessary in order to corectly plot visualisations, and is inferred from the `preprocessor` object below:


```python
numerical_feats_idx  = preprocessor.transformers_[0][2]
categorical_feats_idx  = preprocessor.transformers_[1][2]
scaler = preprocessor.transformers_[0][1].named_steps['scaler']
num_feats_names = [feature_names[i] for i in numerical_feats_idx]
cat_feats_names = [feature_names[i] for i in categorical_feats_idx]
perm_feat_names = num_feats_names + cat_feats_names 
```


```python
pp = pprint.PrettyPrinter()
print("Original order:")
pp.pprint(feature_names)
print("")
print("New features order:")
pp.pprint(perm_feat_names)
```

    Original order:
    ['Age',
     'Workclass',
     'Education',
     'Marital Status',
     'Occupation',
     'Relationship',
     'Race',
     'Sex',
     'Capital Gain',
     'Capital Loss',
     'Hours per week',
     'Country']
    
    New features order:
    ['Age',
     'Capital Gain',
     'Capital Loss',
     'Hours per week',
     'Workclass',
     'Education',
     'Marital Status',
     'Occupation',
     'Relationship',
     'Race',
     'Sex',
     'Country']


Create a utility to reorder the columns of an input array so that the features have the same ordering as that induced by the preprocessor.


```python
def permute_columns(X: np.ndarray, feat_names: List[str], perm_feat_names: List[str]) -> np.ndarray:
    """
    Permutes the original dataset so that its columns (ordered according to feat_names) have the order 
    of the variables after transformation with the sklearn preprocessing pipeline (perm_feat_names).
    """
    
    perm_X = np.zeros_like(X)
    perm = []
    for i, feat_name in enumerate(perm_feat_names):
        feat_idx = feat_names.index(feat_name)
        perm_X[:, i] = X[:, feat_idx]
        perm.append(feat_idx)
    return perm_X, perm
```

The categorical variables will be grouped to reduce shap values variance, as shown in [this](kernel_shap_adult_lr.ipynb) example. To do so, the dimensionality of each categorical variable is extracted from the preprocessor:


```python
# get feature names for the encoded categorical features
ohe = preprocessor.transformers_[1][1].named_steps['onehot']
fts = [feature_names[x] for x in categorical_features]
cat_enc_feat_names = ohe.get_feature_names_out(fts)
 # compute encoded dimension; -1 as ohe is setup with drop='first'
feat_enc_dim = [len(cat_enc) - 1 for cat_enc in ohe.categories_]
d = {'feature_names': fts , 'encoded_dim': feat_enc_dim}
df = pd.DataFrame(data=d)
print(df)
total_dim = df['encoded_dim'].sum() 
print("The dimensionality of the encoded categorical features is {}.".format(total_dim))
assert total_dim == len(cat_enc_feat_names)
```

        feature_names  encoded_dim
    0       Workclass            8
    1       Education            6
    2  Marital Status            3
    3      Occupation            8
    4    Relationship            5
    5            Race            4
    6             Sex            1
    7         Country           10
    The dimensionality of the encoded categorical features is 45.


### Select a subset of test instances to explain


```python
def split_set(X, y, fraction, random_state=0):
    """
    Given a set X, associated labels y, splits a fraction y from X.
    """
    _, X_split, _, y_split = train_test_split(X, 
                                              y, 
                                              test_size=fraction, 
                                              random_state=random_state,
                                             )
    print("Number of records: {}".format(X_split.shape[0]))
    print("Number of class {}: {}".format(0, len(y_split) - y_split.sum()))
    print("Number of class {}: {}".format(1, y_split.sum()))
    
    return X_split, y_split
```


```python
fraction_explained = 0.05 
X_explain, y_explain = split_set(X_test, 
                                 y_test, 
                                 fraction_explained, 
                                 )
X_explain_proc = preprocessor.transform(X_explain)
```

    Number of records: 128
    Number of class 0: 96
    Number of class 1: 32


Create a version of the dataset to be explained that has the same feature ordering as that of the feature matrix after applying the preprocessing (for plotting purposes).


```python
perm_X_explain, _ = permute_columns(X_explain, feature_names, perm_feat_names)
```

## Fit a binary logistic regression classifier to the Adult dataset

### Training 


```python
classifier = LogisticRegression(multi_class='multinomial',
                                random_state=0,
                                max_iter=500,
                                verbose=0,
                               )
classifier.fit(X_train_proc, y_train)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(max_iter=500, multi_class=&#x27;multinomial&#x27;, random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500, multi_class=&#x27;multinomial&#x27;, random_state=0)</pre></div></div></div></div></div>



### Model assessment


```python
y_pred = classifier.predict(X_test_proc)
```


```python
cm = confusion_matrix(y_test, y_pred)
```


```python
title = 'Confusion matrix for the logistic regression classifier'
disp = ConfusionMatrixDisplay.from_estimator(classifier, 
                                             X_test_proc, 
                                             y_test,
                                             display_labels=target_names,
                                             cmap=plt.cm.Blues,
                                             normalize=None,
                                            )
disp.ax_.set_title(title);
```


    
![png](distributed_kernel_shap_adult_lr_files/distributed_kernel_shap_adult_lr_42_0.png)
    



```python
print('Test accuracy: ', accuracy_score(y_test, classifier.predict(X_test_proc)))
```

    Test accuracy:  0.855078125


## Running KernelSHAP in sequential mode

A background dataset is selected. 


```python
start_example_idx = 0
stop_example_idx = 100
background_data = slice(start_example_idx, stop_example_idx)
```

Groups are specified by creating a list where each sublist contains the column indices that a given variable occupies in the preprocessed feature matrix.


```python
def make_groups(num_feats_names: List[str], cat_feats_names: List[str], feat_enc_dim: List[int]) -> Tuple[List[str], List[List[int]]]:
    """
    Given a list with numerical feat. names, categorical feat. names
    and a list specifying the lengths of the encoding for each cat.
    varible, the function outputs a list of group names, and a list
    of the same len where each entry represents the column indices that
    the corresponding categorical feature 
    """
    
    group_names = num_feats_names + cat_feats_names
    groups = []
    cat_var_idx = 0
    
    for name in group_names: 
        if name in num_feats_names:
            groups.append(list(range(len(groups), len(groups) + 1)))
        else:
            start_idx = groups[-1][-1] + 1 if groups else 0
            groups.append(list(range(start_idx, start_idx + feat_enc_dim[cat_var_idx] )))
            cat_var_idx += 1
    
    return group_names, groups
            
def sparse2ndarray(mat, examples=None):
    """
    Converts a scipy.sparse.csr.csr_matrix to a numpy.ndarray.
    If specified, examples is slice object specifying which selects a
    number of rows from mat and converts only the respective slice.
    """
    
    if examples:
        return mat[examples, :].toarray()
    
    return mat.toarray()

X_train_proc_d = sparse2ndarray(X_train_proc, examples=background_data)
group_names, groups = make_groups(num_feats_names, cat_feats_names, feat_enc_dim)
```

Initialise and run the explainer sequentially.


```python
pred_fcn = classifier.predict_proba
seq_lr_explainer = KernelShap(pred_fcn, link='logit', feature_names=perm_feat_names)
seq_lr_explainer.fit(X_train_proc_d[background_data, :], group_names=group_names, groups=groups)
```




    KernelShap(meta={
      'name': 'KernelShap',
      'type': ['blackbox'],
      'task': 'classification',
      'explanations': ['local', 'global'],
      'params': {
                  'link': 'logit',
                  'group_names': ['Age', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'],
                  'grouped': True,
                  'groups': [[0], [1], [2], [3], [4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20], [21, 22, 23, 24, 25, 26, 27, 28], [29, 30, 31, 32, 33], [34, 35, 36, 37], [38], [39, 40, 41, 42, 43, 44, 45, 46, 47, 48]],
                  'weights': None,
                  'summarise_background': False,
                  'summarise_result': None,
                  'transpose': False,
                  'kwargs': {}}
                ,
      'version': '0.7.1dev'}
    )




```python
n_runs = 3
```


```python
s_explanations, s_times = [], []
```


```python
for run in range(n_runs):
    t_start = timer()
    explanation = seq_lr_explainer.explain(sparse2ndarray(X_explain_proc))
    t_elapsed = timer() - t_start
    s_times.append(t_elapsed)
    s_explanations.append(explanation.shap_values)
```

## Running KernelSHAP in distributed mode

The only change needed to distribute the computation is to pass a dictionary containing the number of (physical) CPUs available to distribute the computation to the `KernelShap` constructor:


```python
def distrib_opts_factory(n_cpus: int) -> Dict[str, int]:
    return {'n_cpus': n_cpus}
```


```python
cpu_range = range(2, 5)
distrib_avg_times = dict(zip(cpu_range, [0.0]*len(cpu_range)))
distrib_min_times = dict(zip(cpu_range, [0.0]*len(cpu_range))) 
distrib_max_times = dict(zip(cpu_range, [0.0]*len(cpu_range))) 
d_explanations = defaultdict(list)
```


```python
for n_cpu in cpu_range:
    opts = distrib_opts_factory(n_cpu)
    distrib_lr_explainer = KernelShap(pred_fcn, link='logit', feature_names=perm_feat_names, distributed_opts=opts)
    distrib_lr_explainer.fit(X_train_proc_d[background_data, :], group_names=group_names, groups=groups)
    raw_times = []
    for _ in range(n_runs):
        t_start = timer()
        d_explanations[n_cpu].append(distrib_lr_explainer.explain(sparse2ndarray(X_explain_proc), silent=True).shap_values)
        t_elapsed = timer() - t_start
        raw_times.append(t_elapsed)
    distrib_avg_times[n_cpu] = np.round(np.mean(raw_times), 3)
    distrib_min_times[n_cpu] = np.round(np.min(raw_times), 3)
    distrib_max_times[n_cpu] = np.round(np.max(raw_times), 3)
    ray.shutdown()
```

## Results analysis

### Timing


```python
print(f"Distributed average times for {n_runs} runs (n_cpus: avg_time):")
print(distrib_avg_times)
print("")
print(f"Sequential average time for {n_runs} runs:")
print(np.round(np.mean(s_times), 3), "s")
```

    Distributed average times for 3 runs (n_cpus: avg_time):
    {2: 57.197, 3: 41.728, 4: 36.751}
    
    Sequential average time for 3 runs:
    119.656 s


Running KernelSHAP in a distributed fashion improves the runtime as the results above show. However, the results above should not be interpreted as performance measurements since they were not run in a controlled environment. See our [blog post](https://www.seldon.io/how-seldons-alibi-and-ray-make-model-explainability-easy-and-scalable/) for a more thorough analysis.

### Explanations comparison


```python
cls = 0  # class of prediction explained
run = 1  # which run to compare the result for
```


```python
# sequential
shap.summary_plot(s_explanations[run][cls], perm_X_explain, perm_feat_names)
```


    
![png](distributed_kernel_shap_adult_lr_files/distributed_kernel_shap_adult_lr_65_0.png)
    



```python
# distributed
n_cpu = 3
shap.summary_plot(d_explanations[n_cpu][run][cls], perm_X_explain, perm_feat_names)
```


    
![png](distributed_kernel_shap_adult_lr_files/distributed_kernel_shap_adult_lr_66_0.png)
    


Comparing the results above one sees that the running the algorithm across multiple cores gave identical results, indicating its correctness.

## Conclusion

This example showed that batches of explanations can be explained much faster by simply passing ``distributed_opts={'n_cpus': k}`` to the ``KernelShap`` constructor (here ``k`` is the number of physical cores available). The significant runtime reduction makes it possible to explain larger datasets faster and combine shap values estimated with KernelSHAP into global explanations or use larger background datasets.
