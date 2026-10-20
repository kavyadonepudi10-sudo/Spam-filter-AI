# Spam Filter AI

Most text classifiers are lazy. They trust a default 0.5 threshold and call it a day—missing actual fraud or throwing your real messages straight into the junk folder. 

I built this project to do it right. It's a clean, end-to-end classical machine learning pipeline that doesn't just train models, but dives deep into **decision threshold tuning** to strictly control false positives where it actually matters.

## What's Inside
* **Data:** Automatically fetches the SMS Spam Collection dataset.
* **Features:** TF-IDF vectorization to turn raw text into math (`max_features=5000`).
* **Models Tested:** Naive Bayes (our fast text baseline), Logistic Regression, and Random Forest.
* **The Core Fix:** Precision-recall curve analysis to find an optimal decision boundary instead of guessing.

## Quickstart

Clone it, grab the dependencies, and watch it run:

```bash
git clone [https://github.com/kavyadonepudi10-sudo/Spam-filter-AI.git](https://github.com/kavyadonepudi10-sudo/Spam-filter-AI.git)
cd Spam-filter-AI
pip install -r requirements.txt
python spam_pipeline.py
