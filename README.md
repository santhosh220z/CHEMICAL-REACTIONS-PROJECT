efqefqwefqwefqwefqwe
# CHEMICAL-REACTIONS-PROJECT

---

# ⚗️ Chemical Reaction Success Prediction — AI Powered Web App

This project is a simple, AI-powered web application that predicts the probability of a chemical reaction being successful based on key reaction parameters:

✅ Temperature
✅ pH level
✅ Reactant concentration

The app leverages a trained Artificial Neural Network (ANN) built with TensorFlow & Keras, wrapped in a Flask backend, and presented through a modern, responsive frontend.

---

## 🚀 Features

* 🧪 Predict reaction success probability
* 💻 Web-based user interface (HTML + CSS + JS)
* 📊 Real-time predictions via Flask backend
* 🎨 Clean & responsive design
* 🔐 Simple login & registration system (local storage for demo purposes)
* 📈 ANN model trained on real-world reaction data
* 🧠 Scales input parameters using scikit-learn

---

## 🛠️ Technologies Used

* Python
* Flask
* TensorFlow / Keras
* Pandas
* scikit-learn
* HTML5 / CSS3 / JavaScript

---

## 📂 Project Structure

```
/templates/
    - index.html
    - login.html
    - registration.html
    - about.html
    - contact.html
    
app.py                  (Flask backend)
ann_model_reactions.keras  (trained model)
reactions.csv             (dataset)
README.md
```

---

## ⚙️ How It Works

1. **Dataset**: A CSV dataset `reactions.csv` of past chemical reaction results
2. **Training**: ANN model trained on Temperature, pH, Concentration → Success
3. **Input Scaling**: StandardScaler (scikit-learn) used to normalize inputs
4. **Web Interface**: Users enter parameters through HTML form
5. **Prediction**: Flask backend loads the trained `.keras` model, predicts result
6. **Output**: Success probability and result displayed on the web page

---

## 🧪 Sample Input Parameters (with validation):

| Parameter        | Range           |
| ---------------- | --------------- |
| Temperature (°C) | 0°C – 300°C     |
| pH               | 0 – 14          |
| Concentration    | 0.01 – 10 mol/L |

---

## 🚧 Limitations

* Prototype-level project (for educational/demo use)
* No database — users & data stored in browser localStorage & CSV only
* Accuracy depends on dataset quality
* Not a production-level authentication system

---

## 💡 Why this project?

Chemical reactions are sensitive to various conditions and predicting success can save valuable time and resources in laboratory settings.

This project demonstrates:

* Practical application of AI/ML in chemistry
* Full-stack web app integrating Flask backend & ML model
* Real-time AI-based decision support
* How data science and web development can combine to build useful scientific tools

---

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/3bfd2990-d8a4-429e-a928-bca80c1c7ecd)



## 🎓 Team

Built by **Team 2**
2025 Chemical Reaction Success Prediction — Hackathon Project

---
