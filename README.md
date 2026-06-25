\# Smart Product Pricing Engine



An end-to-end Machine Learning pipeline engineered to predict optimal product prices using unstructured e-commerce catalog content.





\## Project Overview \& Business Problem



Manual pricing leads to lost revenue margins and inconsistent catalog valuations. This project implements a data-driven pricing engine that models how product attributes influence market value.



\* \*\*Objective:\*\* Predict prices for unseen products based entirely on the provided dataset.

\* \*\*Evaluation Metric:\*\* \*\*SMAPE\*\* (Symmetric Mean Absolute Percentage Error).

\---



\## Dataset Description



The model ingests a multi-column dataset containing:



\* `sample-id`: Unique identifier for each product.

\* `catalog content`: Raw text blob containing title, description, quantity, and units.

\* `image link`: URL pointing to the product image resource.

\* `price` \*(Target)\*: Continuous numerical value representing product price.

\---



\## Project Workflow



1\. \*\*Data Preprocessing:\*\* Regex-driven parsing of text blobs into structured features (`item\_name`, `quantity value`, `unit`) and unit normalization.

2\. \*\*Feature Engineering:\*\* Text vectorization via \*\*TF-IDF\*\* followed by \*\*Truncated SVD\*\* dimensionality reduction and metadata extraction.

3\. \*\*Model Training:\*\* Stratified K-Fold cross-validation loops utilizing gradient boosting.

4\. \*\*Inference Pipeline:\*\* Automated transformation of raw test samples into submission-ready outputs.

\---

\## Model \& Tech Stack



\* \*\*Model:\*\* \*\*LightGBM\*\* (Chosen for speed, low memory footprint, and handling of high-dimensional sparse text vectors).

\* \*\*Languages \& Tools:\*\* Python 3, Pandas, NumPy, Scikit-Learn, LightGBM.

\---



\## Key Insights



\* Parsed text features (like extracted quantities) significantly outperformed raw text tokenization alone.

\* Truncated SVD retained over 90% of semantic variance while exponentially speeding up model training.

\---



\## How to Run



1\. \*\*Clone \& Unzip:\*\* Clone the repository and unzip the codebase: `unzip "ML project .zip"`.

2\. \*\*Execute:\*\* Open `ML.ipynb` in Jupyter or Google Colab and run all cells.

3\. \*\*Verify:\*\* Check results in the generated submission file:

```python

import pandas as pd

print(pd.read\_csv('test\_out\_fast.csv').head())



```







```



```

