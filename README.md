# **README — Linear Regression from Scratch with Gradient Descent**

## **Overview**
This project implements **linear regression from scratch** using **gradient descent** and a **matrix formulation** of the model.  
The goal is to demonstrate a solid understanding of the mathematical foundations behind optimization and model training, without relying on machine learning libraries such as scikit‑learn.

The project includes:

- manual implementation of linear regression  
- gradient descent optimization  
- matrix‑based formulation of the model  
- visualization of the weight trajectory in 3D  
- evaluation on a real‑world UCI dataset  
- comparison with scikit‑learn’s implementation  

This exercise is part of my effort to strengthen and showcase my understanding of core ML concepts by implementing fundamental algorithms from first principles.

---

## **Model Formulation**

We model the relationship between two independent variables $\( x_1, x_2 \)$ and a dependent variable y as:

$$\
y_{\text{pred}} = w_1 x_1 + w_2 x_2 + w_3
\$$

Using matrix notation:

$$\
W = 
\begin{bmatrix}
w_1 \\
w_2 \\
w_3
\end{bmatrix},
\quad
X = 
\begin{bmatrix}
x_1 \\
x_2 \\
1
\end{bmatrix}
\$$

$$\
y_{\text{pred}} = W^T X
\$$

---

## **Loss Function**

The cost function is the **Mean Squared Error (MSE)**:

$$\
\text{Loss} = \frac{1}{N} \sum_{j=1}^{N} (y_j - y_{\text{pred},j})^2
\$$

The gradient of the loss with respect to the weights is:

$$\
\nabla_W = -X (y - y_{\text{pred}})
\$$

Weights are updated iteratively using gradient descent:

$$\
W := W - \alpha \nabla_W
\$$

where $\alpha$ is the learning rate.

---

## **Dataset**

The model is tested on the following UCI dataset:
**Forest Fires**  
  - $\ x_1 = \text{wind} \$  
  - $\ x_2 = \text{rain} \$  
  - $\ y = \text{area} \$

---

## **Implementation**

The project includes:

- data loading and preprocessing  
- normalization of features  
- matrix‑based implementation of the model  
- gradient descent loop  
- tracking of weight updates  
- visualization of the 3D trajectory of $\ (w_1, w_2, w_3) \$  
- final evaluation of the model  

All code is written in Python using only:

- NumPy  
- Matplotlib  
- Pandas (for dataset handling)

No ML libraries are used for the core algorithm.

---

## **3D Visualization of Weight Trajectory**

During gradient descent, the weights $\( w_1, w_2, w_3 \)$ evolve over time.  
This project includes a **3D plot** showing the trajectory of the weights across iterations.

This visualization helps illustrate:

- convergence behavior  
- stability of the optimization  
- sensitivity to the learning rate  
- the geometry of the parameter space  

---

## **Comparison with scikit‑learn**

To validate the implementation, the final weights and predictions are compared with:

```python
from sklearn.linear_model import LinearRegression
```

This comparison highlights:

- differences in optimization  
- numerical stability  
- convergence speed  
- accuracy of the manual implementation  

---

## **What I Learned**

Through this project, I strengthened my understanding of:

- the mathematical foundations of linear regression  
- gradient descent optimization  
- matrix formulation of linear models  
- the importance of feature scaling  
- visualization of optimization paths  
- differences between manual and library‑based implementations  

This project also reinforces the importance of understanding ML algorithms beyond simply calling library functions.

---

## **How to Run the Project**

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. Run the notebook or Python script:
   ```bash
   python linear_regression_from_scratch.py
   ```
   or open the `.ipynb` file in Jupyter/Colab.

---

## **Future Improvements**

- Add regularization (Ridge, Lasso)  
- Implement stochastic and mini‑batch gradient descent  
- Extend to polynomial regression  
- Add more visualizations (loss curve, contour plots)  

---

## **License**

This project is released under the MIT License.  
Feel free to explore and learn from the code.
