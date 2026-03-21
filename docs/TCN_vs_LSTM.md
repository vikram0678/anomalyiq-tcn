# TCN vs LSTM — Architectural Comparison

## 1. Parallelization and Training Speed
TCNs use convolutional operations that process the entire
sequence in parallel. LSTMs process tokens sequentially,
making them significantly slower to train.
In this project, TCN training completed in ~120s vs an
estimated ~400s for an equivalent LSTM on the same data.

## 2. Gradient Flow
LSTMs are prone to vanishing/exploding gradients over long
sequences. TCNs use residual connections and stable
convolutional backpropagation, resulting in more stable
and reliable gradient flow during training.

## 3. Receptive Field Management
TCNs use dilated convolutions with exponentially increasing
dilation factors (1, 2, 4, 8...). This gives precise,
predictable control over the receptive field size.
LSTMs rely on hidden state which is harder to interpret
and control for long-range dependency capture.

## 4. Summary Table
| Property              | TCN        | LSTM       |
|-----------------------|------------|------------|
| Training Speed        | Fast       | Slow       |
| Gradient Stability    | High       | Medium     |
| Parallelism           | Yes        | No         |
| Receptive Field       | Controlled | Implicit   |
| Long-range Dependency | Excellent  | Good       |

## Conclusion
For this anomaly detection task on NASA sensor data,
TCN was chosen for its speed, stability, and precise
receptive field control — all critical for production
monitoring systems.