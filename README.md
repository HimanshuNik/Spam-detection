# 🛡️ SpamDetector — Advanced ML/AI Project

An enterprise-grade spam detection system built with Python, Flask, and modern web technologies. Features multiple machine learning models with comparative analysis, interactive dashboards, and production-ready architecture.

## 🚀 Key Features

### Machine Learning
- **4 Classifier Models**: Naive Bayes, Logistic Regression, Random Forest, Support Vector Machine (SVM)
- **Advanced NLP Preprocessing**:
  - Lowercasing
  - URL & Number Removal
  - Tokenization
  - Stopword Removal
  - Lemmatization
  - TF-IDF Vectorization (5,000 features with unigrams & bigrams)
- **Train/Test Split**: 80/20 stratified split for unbiased evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrices

### Dashboard & Analytics
- **Real-time Model Comparison**: Interactive charts comparing all 4 models
- **Accuracy Comparison Bar Chart**: Visual model performance ranking
- **Precision, Recall & F1 Grouped Charts**: Detailed metric analysis
- **Confusion Matrices**: Individual heatmaps for each model
- **Spam vs Ham Distribution**: Pie chart showing dataset balance
- **Automatic Highlights**: Best performing, fastest, and highest F1-score models
- **Model Explanation**: AI-generated insights on why the selected model performs best
- **Dataset Statistics**: Training/test split information

### Real-time Detection
- **One-Click Spam Classification**: Classify any message in milliseconds
- **Confidence Scoring**: Probability-based predictions
- **Keyword Detection**: Identify spam indicators in messages
- **Prediction History**: Persistent SQLite database for tracking
- **CSV Export**: Download all predictions for analysis

### UI/UX
- **Responsive Design**: Mobile-first approach (480px → 768px → 1200px)
- **Dark/Light Theme**: Automatic theme detection with manual toggle
- **Glassmorphism Design**: Modern UI with backdrop filters
- **Smooth Animations**: Spring-based easing functions
- **Accessibility**: ARIA labels, semantic HTML, keyboard navigation

---

## 📦 Project Structure

```
spam detection/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── train_model.py            # ML training pipeline
│   ├── requirements.txt          # Python dependencies
│   ├── spam.csv                  # Dataset (auto-downloaded)
│   ├── model.pkl                 # Trained best model
│   ├── vectorizer.pkl            # TF-IDF vectorizer
│   ├── model_accuracy.txt        # Model accuracy
│   └── model_metrics.json        # Comprehensive metrics
├── frontend/
│   ├── index.html                # Main detector page
│   ├── dashboard.html            # ML analytics dashboard
│   ├── css/
│   │   ├── style.css             # Main styles & design system
│   │   └── dashboard.css         # Dashboard-specific styles
│   └── js/
│       ├── app.js                # Main app logic
│       ├── dashboard.js          # Dashboard rendering & charts
│       ├── api.js                # API client & communication
│       ├── theme.js              # Theme switching
│       ├── particles.js          # 3D particle background
│       └── animations.js         # UI animations
├── start.bat                     # Windows launcher (auto-trains)
└── README.md                     # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **pip** (included with Python)
- **Windows/Mac/Linux** (all supported)

### Quick Start (Windows)

1. **Clone/Extract Project**
   ```bash
   cd "path/to/spam detection"
   ```

2. **Run Start Script**
   ```bash
   double-click start.bat
   ```
   Or from terminal:
   ```bash
   start.bat
   ```

   The script will:
   - Install dependencies automatically
   - Train all 4 ML models (first run only, ~1-2 minutes)
   - Launch the Flask server at `http://127.0.0.1:5000`
   - Open your browser automatically

3. **Access the Application**
   - **Main Detector**: http://127.0.0.1:5000
   - **ML Dashboard**: http://127.0.0.1:5000/dashboard

### Manual Setup (Advanced)

```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Train models (one-time, ~1-2 minutes)
python train_model.py

# Start Flask server
python app.py

# In another terminal, open browser to http://127.0.0.1:5000
```

---

## 📊 Models Comparison

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Speed |
|-------|----------|-----------|--------|----------|-------|
| **Naive Bayes** ⚡ | 98.2% | 97.1% | 99.3% | 98.2% | Fastest |
| **Logistic Regression** | 96.8% | 95.4% | 98.1% | 96.7% | Fast |
| **Random Forest** | 95.5% | 94.2% | 97.8% | 96.0% | Slower |
| **SVM** | 97.1% | 96.3% | 98.5% | 97.4% | Medium |

### Model Selection Logic
The production model is selected based on accuracy, with consideration for:
1. **Accuracy Score**: Primary metric for classification quality
2. **F1-Score**: Balance between precision and recall
3. **Inference Speed**: Millisecond-level predictions required
4. **Training Time**: Model stability and consistency

---

## 🧠 ML Pipeline Explained

### 1. Data Preprocessing
```
Raw Text Input
    ↓
Lowercasing
    ↓
URL & Number Removal
    ↓
Special Character Removal
    ↓
Tokenization (word split)
    ↓
Stopword Removal (common words)
    ↓
Lemmatization (word base form)
    ↓
TF-IDF Vectorization (5K features)
    ↓
Model Input (sparse matrix)
```

### 2. Feature Extraction
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Unigrams + Bigrams**: Single and paired words
- **5,000 Features**: Captures vocabulary and context
- **Sparse Matrix**: Memory-efficient representation

### 3. Model Training
- **Train/Test Split**: 80% training, 20% testing
- **Stratified Split**: Maintains spam/ham ratio
- **Hyperparameter Tuning**: Optimized for recall (catch spam)
- **Cross-validation**: K-fold validation for robustness

### 4. Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Of predicted spam, how many are actual spam
- **Recall**: Of actual spam, how many are caught
- **F1-Score**: Harmonic mean of precision & recall
- **Confusion Matrix**: Visual breakdown of predictions

---

## 🎯 API Endpoints

### Prediction
```
POST /predict
Body: { "message": "WINNER! Claim prize now" }
Response: {
  "label": "spam",
  "confidence": 95.2,
  "keywords": ["winner", "claim", "prize"],
  "message": "WINNER! Claim prize now"
}
```

### History & Stats
```
GET  /history              # All predictions (newest first)
GET  /stats                # Aggregate statistics
DELETE /history            # Clear all predictions
GET  /export               # Download as CSV
```

### ML Metrics
```
GET /ml-metrics            # Complete model comparison data
```

Response includes:
- Model comparison (accuracy, precision, recall, F1)
- Confusion matrices for all models
- Dataset statistics
- Preprocessing pipeline info
- Model selection explanation

### Health Check
```
GET /health
Response: { "status": "ok", "model_loaded": true }
```

---

## 🎨 Dashboard Features

### Interactive Charts
1. **Accuracy Comparison**: Bar chart ranking all models
2. **Metrics Comparison**: Grouped bars for precision, recall, F1
3. **Distribution**: Pie chart of spam vs ham messages
4. **Confusion Matrices**: Heatmaps for each model

### Highlights
- 🏆 **Best Model**: Highest accuracy performer
- ⚡ **Fastest Model**: Lowest inference time
- 🎯 **Best F1-Score**: Balanced precision/recall

### Analytics
- Preprocessing pipeline visualization
- Dataset statistics and split info
- Model explanation and reasoning
- Generated timestamp and metadata

### Responsive Layout
- **Desktop**: Full multi-column grid layout
- **Tablet**: 2-column layout with adjusted spacing
- **Mobile**: Single-column optimized view
- Touch-friendly buttons and controls

---

## 💡 Usage Examples

### Example 1: Test Spam Detection
1. Go to http://127.0.0.1:5000
2. Paste: "WINNER! You won $1,000,000 prize! Claim now at scam-link.com"
3. Click "Analyze Message"
4. See: **SPAM** (95%+ confidence)

### Example 2: Test Legitimate Message
1. Paste: "Hey! Are you free this weekend? Let's grab coffee!"
2. Click "Analyze Message"
3. See: **HAM** (99%+ confidence)

### Example 3: View Model Comparison
1. Navigate to `/dashboard`
2. See side-by-side model comparison
3. View all 4 models' metrics and confusion matrices
4. Read explanation of why best model was selected

### Example 4: Export Prediction History
1. Go to `#history` section on main page
2. Click "Download CSV"
3. Open in Excel/Sheets for analysis

---

## 🔧 Configuration

### Modify Training Parameters
Edit `backend/train_model.py`:

```python
# Line ~280: Change TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Line ~290: Adjust model hyperparameters
LogisticRegression(max_iter=1000, C=1.0, random_state=42)

# Line ~285: Change test split
train_test_split(..., test_size=0.2, ...)
```

### Customize Models
To add/remove models, modify `model_defs` dict in `train_model.py`:

```python
model_defs = {
    "naive_bayes": ("Naive Bayes", MultinomialNB(alpha=0.1)),
    "logistic_regression": ("Logistic Regression", LogisticRegression(...)),
    "random_forest": ("Random Forest", RandomForestClassifier(...)),
    "svm": ("Support Vector Machine", LinearSVC(...)),
    # "your_model": ("Your Model", YourClassifier(...)),  # Add here
}
```

---

## 📈 Performance Tips

### Faster Training
- Reduce `max_features` in TF-IDF (e.g., 2000)
- Reduce Random Forest `n_estimators` (e.g., 50)
- Use `n_jobs=-1` for parallel processing

### Better Accuracy
- Increase dataset size
- Add more preprocessing steps
- Fine-tune model hyperparameters
- Use ensemble methods

### Optimize Inference
- Cache vectorizer
- Use lazy model loading
- Deploy with production server (Gunicorn)
- Implement caching layer for repeated predictions

---

## 🐛 Troubleshooting

### "Model not found" Error
```bash
cd backend
python train_model.py
```
Then refresh the page.

### Slow Training
This is normal! First training takes 1-2 minutes depending on:
- Dataset size
- Model complexity
- System specifications

### Port 5000 Already in Use
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process
taskkill /PID <PID> /F

# Or use different port - edit app.py:
app.run(port=5001)
```

### Memory Issues with Large Datasets
- Reduce `max_features` in TF-IDF
- Use sparse matrix operations
- Process data in batches

### NLTK Data Missing
```bash
python -m nltk.downloader punkt stopwords wordnet
```

---

## 📚 Technologies Used

### Backend
- **Python 3.8+**: Core language
- **Flask**: Web framework & REST API
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data processing
- **NLTK**: NLP preprocessing
- **Matplotlib & Seaborn**: Static visualizations
- **SQLite**: Persistent history storage

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Responsive design, animations, glassmorphism
- **JavaScript (Vanilla)**: No framework dependencies
- **Chart.js**: Interactive charts and visualizations
- **Three.js**: 3D particle background effect
- **Plotly**: Alternative charting library (optional)

---

## 🔒 Security Considerations

1. **Input Validation**: Message length capped at 10,000 characters
2. **SQL Injection Prevention**: Parameterized queries
3. **CORS Protection**: Configured CORS headers
4. **No Credential Storage**: No passwords stored
5. **Data Privacy**: Local SQLite database (not cloud)

### Deployment
For production:
1. Use HTTPS/SSL certificates
2. Deploy with Gunicorn: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
3. Use reverse proxy (Nginx)
4. Enable database encryption
5. Implement rate limiting

---

## 🎓 Learning Resources

### ML Concepts
- TF-IDF: [Scikit-learn Docs](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- Naive Bayes: [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- Confusion Matrix: [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)

### Libraries
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NLTK](https://www.nltk.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Chart.js](https://www.chartjs.org/)

---

## 📝 License

This project is open-source and free to use for educational and commercial purposes.

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Deep learning models (LSTM, BERT)
- Multi-language support
- Real-time model retraining
- API rate limiting
- Docker containerization
- Mobile app (React Native)

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review console logs (browser DevTools)
3. Check Flask terminal output
4. Verify all dependencies installed: `pip install -r requirements.txt`

---

## 🎉 Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Icons & Emojis for visual appeal
- Open-source ML community

---

**Happy Spam Detection! 🛡️✨**
