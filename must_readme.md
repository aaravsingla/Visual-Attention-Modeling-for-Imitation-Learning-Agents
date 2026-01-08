# Visual Attention Modeling for Imitation Learning Agents

[![Open In Colab](https://colab.research.google.com/drive/1QHAwl5OD2iJ7zUF04dKCR6-dSKawgNfR?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Stable-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Abstract
This research project explores the intersection of **Computational Interaction**, **Computer Vision**, and **Imitation Learning** to model human reading dynamics. By formulating the reading process as a sequential decision-making problem (Markov Decision Process), we simulate how a rational agent allocates visual attention based on text complexity.

The core objective is to validate the **Eye-Mind Hypothesis** via machine learning: demonstrating that an artificial agent can learn to correlate **dwell time** (fixation duration) with **processing cost** (word difficulty) purely by observing expert traces, without explicit rule-based programming. This serves as a foundational step for building generative user models that can predict human visual attention in real-time.

## 🧠 Theoretical Framework
The project simulates "Reading" not as a text-parsing task, but as an **Oculomotor Control task**:
* **The Environment:** A sentence is treated as a 1D visual field where each token has an associated "Cognitive Load" (0.0 to 1.0).
* **The Agent:** A foveated vision model that must decide *where* to look and *how long* to look there.
* **The Objective:** To minimize the divergence between the agent's scan-path and a human expert's scan-path.

## 🔬 Methodology

### 1. The Cognitive Environment (MDP)
We utilize a custom OpenAI Gym-style environment (`ReadingEnv`) characterized by:
-   **State Space ($S$):** A continuous vector $[p, c]$, where $p$ is the normalized position in the sequence and $c$ is the visual complexity of the current focal point.
-   **Action Space ($A$):** Discrete motor commands:
    -   `0` (Fixate): Maintain gaze to process information (Dwell).
    -   `1` (Saccade): Shift visual attention to the next token.

### 2. Expert Data Generation
To train the agent, we generate synthetic "expert" data based on established psycholinguistic patterns:
> *Probability of fixation increases non-linearly with word complexity.*

### 3. Policy Training (Behavior Cloning)
We employ **Behavior Cloning (BC)**, a supervised imitation learning technique. A neural network policy $\pi_\theta$ is optimized to map states to actions by minimizing the cross-entropy loss between the predicted action distribution and the expert's empirical distribution:

$$\mathcal{L}(\theta) = -\mathbb{E}_{(s,a) \sim \pi_{expert}} [\log \pi_\theta(a|s)]$$

## 📊 Results & Visualization

The agent successfully converges to a policy that mimics the "cognitive load" strategy of the expert. The heatmap below illustrates the alignment between the **Ground Truth Difficulty** (Top Row) and the **Agent's Learned Dwell Time** (Bottom Row).

![Visual Attention Heatmap](results_heatmap.png)

* **Observation:** Darker regions in the "Word Difficulty" row (indicating complex terms) correspond to darker regions in the "Agent Dwell Time" row.
* **Conclusion:** The agent has learned to "slow down" and "read carefully" when encountering complex visual features, validating the efficacy of Imitation Learning for oculomotor simulation.

## 🛠️ Usage

To reproduce the experiments and visualizations:

1.  **Clone the repository:**
    ```bash
    git clone [address link]
    cd visual-attention-imitation-rl
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Simulation:**
    You can run the experiment directly via the provided Jupyter Notebook:
    ```bash
    jupyter notebook Visual_Attention_Imitation_Model.ipynb
    ```

## 🚀 Future Scope
This project lays the groundwork for more advanced User Modelling applications:
1.  **Inverse Reinforcement Learning (IRL):** Recovering the implicit reward function of reading (e.g., the trade-off between speed and comprehension).
2.  **Real-World Data:** Training the model on the **GazeBase** or **EMNLP** eye-tracking datasets.
3.  **Generative UI Testing:** Using the trained agent to evaluate the readability of User Interface designs automatically.

## 👤 Author
**Aarav Singla**
*MS Research Student*
*Specialization: Computer Vision & Reinforcement Learning*

---
*Developed as a precursor to Project 2302: Simulating how people read with RL.*