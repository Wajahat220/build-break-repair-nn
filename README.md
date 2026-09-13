# Fashion-MNIST MLP: Building, Breaking, and Fixing a Neural Network

This repository contains a Kaggle notebook implementing a full deep learning workflow on
Fashion-MNIST: a from-scratch NumPy backpropagation implementation, an activation and
loss-function study, an optimizer comparison, deliberate overfitting, a regularization
study, and a final tuned model selected via cross-validated random search.

## Final Result

| Metric | Value |
|---|---|
| Part 2 baseline test accuracy | 83.43% |
| Final model test accuracy | **89.14%** |
| Improvement | **+5.71 percentage points** |
| Final architecture | 784 → 128 → 64 → 10 |
| Final optimizer | Adam, lr=0.001, dropout=0.1 |

See `DLP_Final_Summary.docx` for the one-page summary.

## Repository Contents

- `fashion_mnist_notebook.ipynb` — the complete notebook (Parts 1–7)
- `DLP_Final_Summary.docx` — one-page final results summary
- `README.md` — this file

## Dataset

[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist), loaded
directly from the CSV files (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`) — not
the IDX format.

## How to Reproduce

1. Open a new Kaggle Notebook.
2. Under **Add Data**, attach the `zalando-research/fashionmnist` dataset.
3. Under **Settings → Accelerator**, select **GPU T4 x2**.
4. Upload or copy in `fashion_mnist_notebook.ipynb`.
5. Run all cells top to bottom (**Run All**). Cells must be run in order — later
   sections reuse variables, models, and helper functions defined earlier (for example,
   Part 6 depends on `OverfitMLP` and the training subset defined in Part 5).
6. Total runtime is dominated by Part 7's random search, which trains 12 hyperparameter
   configurations under 5-fold cross-validation (60 training runs total) — expect this
   cell alone to take a significant portion of the notebook's total runtime on a T4.

All results are reproducible with `SEED = 42`, set for Python's `random`, NumPy,
PyTorch, and CUDA at the start of the notebook.

## Notebook Structure

| Part | Contents |
|---|---|
| Setup | Reproducibility, data loading, 80/20 stratified train/validation split |
| Part 1 | NumPy MLP from scratch (784→64→10), manual backprop, verified against PyTorch autograd |
| Part 2 | Activation function comparison (sigmoid, tanh, ReLU, leaky ReLU), gradient analysis, dead-ReLU check |
| Part 3 | Cross-entropy vs. MSE comparison, separate California Housing regression experiment |
| Part 4 | Optimizer comparison (SGD, SGD+momentum, RMSprop, Adam), same-LR and tuned-LR experiments |
| Part 5 | Deliberate overfitting: 2000 samples, 4×512-unit network, trained past 99% training accuracy |
| Part 6 | Regularization study: L2 (3 values), L1, dropout (3 rates), batch norm, early stopping, data augmentation, more training data |
| Part 7 | Random search (12 configs) scored by 5-fold cross-validation, final test evaluation, confusion matrix, per-class metrics |

## Key Findings

- Manual NumPy backpropagation matched PyTorch autograd to within 3.2 × 10⁻⁸ — confirming the manual implementation is correct.
- Sigmoid showed clear vanishing-gradient behavior compared to ReLU/tanh/leaky ReLU under identical conditions.
- Adam reached 85% validation accuracy in 2 epochs at lr=0.01, while plain SGD never reached it at the same learning rate — the single most impactful design choice in the whole project.
- Forcing overfitting on a 2000-sample subset with an oversized network produced a stable, reproducible 16–18 percentage point generalization gap.
- Among regularization techniques tested on the overfit subset, more training data gave the largest genuine improvement without hurting training accuracy; dropout (p=0.2) was the strongest technique that didn't rely on adding data.
- Scoring random search by 5-fold cross-validation (rather than a single validation split) produced a final model whose test accuracy matched its cross-validated estimate — validating why cross-validation was required rather than a simpler holdout approach.

## Limitations

- Random search covered 12 of many possible hyperparameter configurations under a fixed epoch budget; a larger or longer search may find further improvements.
- Fashion-MNIST is a relatively simple, balanced, low-resolution dataset compared to real-world image classification tasks — results here should not be assumed to transfer directly to more complex data.
- Validation performance was reused across several rounds of experimentation (Part 6 and Part 7), which carries some risk of mild overfitting to the particular validation split, despite cross-validation being used specifically to reduce this risk in Part 7.

## Requirements

- Python 3, PyTorch, NumPy, pandas, scikit-learn, matplotlib
- Kaggle Notebook environment with GPU T4 x2 (or any CUDA-capable GPU)
