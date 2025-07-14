# reddit_persona_generator

Reddit User Persona Generator is a Flask-based web application that creates detailed personas from Reddit user activity.
It analyzes comments and posts to infer demographics, personality traits, interests, and sentiment.
The app uses PRAW (Python Reddit API Wrapper), NLP techniques, and data visualization for insights.
Each persona includes citations for transparency and can be accessed via a unique session ID.

# Tech Stack

- Python  
- Flask  
- PRAW (Python Reddit API Wrapper)  
- TextBlob  
- Matplotlib & Seaborn  
- WordCloud  
- Jinja2 Templates

# Features
 Demographic Inference — Estimates age, gender, and location based on post/comment content.
 Personality Profiling — Identifies top personality traits and sentiment tendencies.
 Visual Insights — Generates charts for activity patterns and sentiment distribution.
 Citation-Backed Insights — All persona attributes are supported by actual Reddit content snippets.
 Interest Detection — Finds user's favorite subreddits and categorizes interests and hobbies.
 Behavior Timeline & Activity Heatmap — Analyzes when and how often users post or comment.
 Async Persona Generation — Supports real-time persona generation via background threading.

# Output
<img width="799" height="628" alt="Screenshot 2025-07-14 195044" src="https://github.com/user-attachments/assets/d8713a63-f724-4b41-b0ba-46bd9a3afe5d" />

# Installation
git clone https://github.com/yourusername/reddit-persona-generator.git
cd reddit-persona-generator
pip install -r requirements.txt
python app.py


