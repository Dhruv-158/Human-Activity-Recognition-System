
## Human Activity Recognition (HAR) System using Machine Learning

### In Simple Words

This project is an AI-based system that can detect and classify what physical activity a person is performing (such as walking, sitting, or standing) using sensor data collected from a smartphone.

---

### What the Project Does

The model uses motion sensor values from a mobile device, primarily:

- **Accelerometer** → detects movement and acceleration  
- **Gyroscope** → detects rotational motion and body orientation  

The collected data is processed and fed into machine learning models, which then predict the activity being performed.

---

### Example of Behavior

| Sensor Data Pattern | Model Prediction |
|--------------------|------------------|
| High motion + repeating leg pattern | Walking |
| Stable signals + very low movement | Standing |
| Upward movement pattern with step variation | Walking Upstairs |

---

### Activities This Model Can Recognize

The model is capable of classifying the following **6 human activities:**

- Walking  
- Walking Upstairs  
- Walking Downstairs  
- Sitting  
- Standing  
- Laying  

---

### Models Built and Compared

| Model Type | Purpose | Final Accuracy |
|------------|---------|---------------|
| Random Forest | Baseline classical machine-learning model | 92.77% |
| Support Vector Machine (SVM) | Best performing classical model | **95.52%** |
| Neural Network (TensorFlow) | Deep learning approach | 92.94% |

**Best Model:** Support Vector Machine (SVM)

---

### Why This Project Is Important

Human Activity Recognition is widely used in real-world applications such as:

- Fitness and workout tracking
- Health and elderly monitoring
- Smartwatches (Apple Watch, Fitbit, Garmin)
- Rehabilitation support
- Smart home automation and IoT systems

This makes the project not just a technical exercise, but a system with strong industry relevance and practical applications.

---
