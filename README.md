# SVM_from_scratch
Support Vector Machine (soft margin) with evaluation and visualisation in python using only pandas and numby libraries.

# Support Vector Machine (Soft Margin) from Scratch

This project provides a custom, from-scratch implementation of a linear Support Vector Machine (SVM) utilizing a soft margin classifier. It relies on fundamental Python matrix operations using the `numpy` library rather than high-level machine learning frameworks.

## Key Features

* **Data Preprocessing:** Automatically applies Z-score standardization to the input features to ensure stable and efficient convergence during gradient descent.
* **Custom Optimization:** Uses a custom gradient descent loop equipped with a convergence threshold to minimize the objective function.
* **Soft Margin Classification:** Incorporates hinge loss to handle non-linearly separable data by penalizing misclassifications and margin violations, balanced by an L2 regularization parameter.
* **Decision Visualization:** Generates a 2D contour plot mapping the decision boundary and the exact margin lines against the standardized dataset.

## Mathematical Background

The model optimizes the weight vector ($w$) and the bias term ($b$) by minimizing a cost function that combines an L2 regularization term and the mean hinge loss:

$$J(w, b) = \lambda \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i (w \cdot x_i - b))$$

During the gradient descent step, the gradients are evaluated for each data point $i$:

* If $y_i (w \cdot x_i - b) \ge 1$ (the point is correctly classified and outside the margin):
    $$\frac{\partial J_i}{\partial w} = 2 \lambda w$$
    $$\frac{\partial J_i}{\partial b} = 0$$

* If $y_i (w \cdot x_i - b) < 1$ (the point violates the margin or is misclassified):
    $$\frac{\partial J_i}{\partial w} = 2 \lambda w - y_i x_i$$
    $$\frac{\partial J_i}{\partial b} = y_i$$

These gradients are accumulated and averaged over the entire dataset of size $n$ before applying the learning rate to update the parameters.

## File Structure

* `SVM_from_scratch_soft_margin.py`: The core script containing the SVM logic, gradient descent algorithm, and visualization code.
* `student_data_svm.csv`: The dataset required to run the script. It must include the columns `math_test`, `lang_test`, and the binary target variable `got_in`.

## Requirements and Installation

To execute this script, ensure you have Python installed along with the following libraries:

* `numpy`
* `pandas`
* `matplotlib`

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
