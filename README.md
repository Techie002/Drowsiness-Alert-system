# Drowsiness-Alert-system
A Python-based application that detects driver drowsiness using computer vision and alerts the driver with an alarm sound to prevent accidents.

---

## 📌 Features
- Real-time eye detection using OpenCV and Haar cascades.
- Monitors eye aspect ratio to detect drowsiness.
- Plays an alarm (`alarm.wav`) when drowsiness is detected.
- Simple Flask web interface with `templates/` and `static/` support.

---

## 🛠️ Tech Stack
- **Python 3.x**
- **OpenCV** for image processing
- **Flask** for web interface
- **Pygame / playsound** for audio alerts


## 📂 Project Structure
│── app.py              # Main application script
│── templates/          # HTML templates for Flask
│── static/             # CSS, JS, and image files
│── alarm.wav           # Alert sound file
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
## Create a virtual environment
- python -m venv venv
- source venv/bin/activate   # On Linux/Mac
- venv\Scripts\activate      # On Windows
## Install dependencies
- pip install -r requirements.txt
## Run the application
- python app.py

