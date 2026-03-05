# 📩 SMS Spam Detection AI

An AI-powered web application that detects whether an SMS message is **Spam** or **Not Spam** using Natural Language Processing and Machine Learning.

The system processes raw text messages, converts them into numerical features using **TF-IDF vectorization**, and classifies them using a trained **Logistic Regression model**. A simple interactive interface allows users to test messages in real time.

---

# 🚀 Live Demo

Try the live application:

https://kethonyx-spam-detector.hf.space

Example:

Input:
```
free prize claim now
```

Output:
```
Spam (confidence: 0.94)
```

---

# 🧠 How It Works

The project implements a simple NLP classification pipeline:

```
User Input
      ↓
Text Preprocessing
(lowercase + punctuation removal)
      ↓
TF-IDF Vectorization
      ↓
Machine Learning Model
(Logistic Regression)
      ↓
Spam / Not Spam Prediction
      ↓
Web Interface (Gradio)
```

---

# ✨ Features

- NLP text preprocessing pipeline  
- TF-IDF feature extraction  
- Model comparison (Naive Bayes vs Logistic Regression)  
- Spam probability estimation  
- Confusion matrix and classification metrics  
- Interactive web interface for real-time predictions  
- Deployed online for easy testing  

---

# 🛠 Technologies Used

- Python  
- scikit-learn  
- pandas  
- NumPy  
- Gradio  

Machine Learning methods used:

- TF-IDF Vectorization  
- Logistic Regression  
- Naive Bayes  
- Regex-based text preprocessing  

---

# 📂 Project Structure

```
spam-detector
│
├── app.py
├── requirements.txt
├── README.md
│
└── data
    └── spam.csv
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python app.py
```

The web interface will start locally and open in your browser.

---

# 🧪 Example Test Messages

Spam examples:

```
free prize claim now
win free iphone today
urgent call this number now
claim your free reward
```

Normal messages:

```
hey how are you
call me when you arrive
let's meet tomorrow
did you finish the assignment
```

---

# 📊 Model Performance

The model is evaluated using standard machine learning metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The model achieves high accuracy on the SMS Spam dataset while maintaining good recall for spam messages.

---

# 📚 Dataset

This project uses the **SMS Spam Collection Dataset**, a public dataset containing labeled SMS messages categorized as spam or legitimate.

Dataset size:
- 5572 SMS messages
- Two classes: Spam and Ham (Not Spam)

---

# 👨‍💻 Author

Dimash Sailau  

Machine Learning enthusiast interested in building intelligent systems using NLP and modern AI tools.

---

# 📌 Future Improvements

Possible improvements for the project:

- Transformer-based models (BERT / DistilBERT)
- Larger training datasets
- Advanced text preprocessing
- API deployment for integration into messaging platforms
- Model optimization and hyperparameter tuning

---

⭐ If you found this project useful, feel free to star the repository!