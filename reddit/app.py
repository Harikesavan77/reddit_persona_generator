#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify, send_file
import praw
import json
import re
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import time
from typing import Dict, List, Any, Tuple
import statistics
from dataclasses import dataclass, asdict
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web deployment
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from pathlib import Path
import io
import base64
import threading
import uuid

app = Flask(__name__)

# Configuration for Reddit API
REDDIT_CONFIG = {
    'client_id': 'LFCyxhkuEm71fAduHhDAgA',
    'client_secret': '18s9w2JCCJJvrsHOGqcx7P_53OENDw',
    'user_agent': 'PersonaGenerator/1.0 by Nervous-Macaron-5971'
}

# Global storage for analysis results (in production, use Redis or database)
analysis_results = {}
analysis_status = {}

@dataclass
class Citation:
    """Represents a citation for persona characteristics"""
    content: str
    url: str
    timestamp: datetime
    type: str  # 'comment' or 'post'
    subreddit: str
    score: int
    confidence: float

@dataclass
class UserPersona:
    """Comprehensive user persona structure"""
    username: str
    account_age: int
    total_karma: int
    
    # Demographics & Basic Info
    estimated_age_range: str
    likely_gender: str
    estimated_location: str
    occupation_indicators: List[str]
    
    # Personality & Behavior
    personality_traits: List[str]
    communication_style: str
    humor_style: str
    political_leaning: str
    
    # Interests & Hobbies
    primary_interests: List[str]
    secondary_interests: List[str]
    expertise_areas: List[str]
    
    # Digital Behavior
    activity_pattern: Dict[str, Any]
    favorite_subreddits: List[str]
    engagement_style: str
    content_preference: str
    
    # Unique Insights
    sentiment_profile: Dict[str, float]
    linguistic_patterns: Dict[str, Any]
    behavioral_timeline: List[Dict[str, Any]]
    influence_network: Dict[str, int]
    
    # Meta Information
    analysis_confidence: float
    generation_timestamp: datetime
    citations: Dict[str, List[Citation]]

class RedditPersonaGenerator:
    def __init__(self, reddit_config: Dict[str, str]):
        """Initialize the persona generator with Reddit API credentials"""
        self.reddit = praw.Reddit(**reddit_config)
        self.patterns = self._load_analysis_patterns()
        self.sentiment_keywords = self._load_sentiment_keywords()
        
    def _load_analysis_patterns(self) -> Dict[str, Any]:
        """Load patterns for analyzing user characteristics"""
        return {
            'age_indicators': {
                'teen': ['high school', 'homework', 'parents', 'teenager', 'teen'],
                'young_adult': ['college', 'university', 'student', 'graduation', 'dorm'],
                'adult': ['work', 'job', 'career', 'mortgage', 'marriage', 'kids'],
                'middle_aged': ['retirement', 'pension', 'grandkids', 'back pain', 'midlife'],
                'senior': ['retired', 'grandchildren', 'medicare', 'social security']
            },
            'gender_indicators': {
                'male': ['bro', 'dude', 'guy', 'man', 'male', 'boyfriend', 'husband', 'father', 'dad'],
                'female': ['girl', 'woman', 'female', 'girlfriend', 'wife', 'mother', 'mom', 'daughter']
            },
            'personality_traits': {
                'extroverted': ['party', 'social', 'friends', 'outgoing', 'crowd'],
                'introverted': ['alone', 'quiet', 'introvert', 'solitude', 'shy'],
                'analytical': ['analyze', 'data', 'logic', 'reasoning', 'systematic'],
                'creative': ['art', 'creative', 'design', 'music', 'writing'],
                'optimistic': ['positive', 'hopeful', 'bright', 'good', 'amazing'],
                'pessimistic': ['negative', 'hopeless', 'bad', 'terrible', 'awful']
            },
            'occupation_indicators': {
                'tech': ['programming', 'coding', 'software', 'developer', 'engineer', 'IT'],
                'healthcare': ['doctor', 'nurse', 'medical', 'hospital', 'patient'],
                'education': ['teacher', 'professor', 'student', 'school', 'education'],
                'finance': ['bank', 'investment', 'finance', 'money', 'trading'],
                'creative': ['artist', 'designer', 'writer', 'musician', 'creative']
            }
        }
    
    def _load_sentiment_keywords(self) -> Dict[str, List[str]]:
        """Load sentiment analysis keywords"""
        return {
            'positive': ['love', 'great', 'awesome', 'amazing', 'excellent', 'fantastic', 'wonderful'],
            'negative': ['hate', 'terrible', 'awful', 'horrible', 'disgusting', 'annoying', 'stupid'],
            'neutral': ['okay', 'fine', 'average', 'normal', 'standard', 'typical']
        }
    
    def extract_user_data(self, username: str) -> Dict[str, Any]:
        """Extract comprehensive user data from Reddit"""
        try:
            user = self.reddit.redditor(username)
            
            # Get user info
            user_info = {
                'username': username,
                'account_created': datetime.fromtimestamp(user.created_utc),
                'comment_karma': user.comment_karma,
                'link_karma': user.link_karma,
                'total_karma': user.comment_karma + user.link_karma,
                'is_verified': user.verified if hasattr(user, 'verified') else False
            }
            
            # Extract posts and comments
            posts = []
            comments = []
            
            for post in user.submissions.new(limit=100):
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'content': post.selftext,
                    'url': f"https://reddit.com{post.permalink}",
                    'subreddit': post.subreddit.display_name,
                    'timestamp': datetime.fromtimestamp(post.created_utc),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'upvote_ratio': post.upvote_ratio
                })
            
            for comment in user.comments.new(limit=200):
                comments.append({
                    'id': comment.id,
                    'content': comment.body,
                    'url': f"https://reddit.com{comment.permalink}",
                    'subreddit': comment.subreddit.display_name,
                    'timestamp': datetime.fromtimestamp(comment.created_utc),
                    'score': comment.score,
                    'parent_id': comment.parent_id
                })
            
            user_info['posts'] = posts
            user_info['comments'] = comments
            
            return user_info
            
        except Exception as e:
            print(f"Error extracting data for {username}: {str(e)}")
            return None
    
    def analyze_demographics(self, user_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[Citation]]]:
        """Analyze user demographics with citations"""
        demographics = {}
        citations = defaultdict(list)
        
        all_text = []
        all_content = []
        
        # Combine all text content
        for post in user_data['posts']:
            content = f"{post['title']} {post['content']}"
            all_text.append(content.lower())
            all_content.append(('post', post, content))
        
        for comment in user_data['comments']:
            content = comment['content']
            all_text.append(content.lower())
            all_content.append(('comment', comment, content))
        
        combined_text = ' '.join(all_text)
        
        # Age analysis
        age_scores = {}
        for age_group, keywords in self.patterns['age_indicators'].items():
            score = sum(combined_text.count(keyword) for keyword in keywords)
            age_scores[age_group] = score
        
        if age_scores:
            likely_age = max(age_scores, key=age_scores.get)
            demographics['estimated_age_range'] = likely_age
            
            # Find citations for age
            for content_type, item, content in all_content:
                for keyword in self.patterns['age_indicators'][likely_age]:
                    if keyword in content.lower():
                        citations['estimated_age_range'].append(
                            Citation(
                                content=content[:200] + "...",
                                url=item['url'],
                                timestamp=item['timestamp'],
                                type=content_type,
                                subreddit=item['subreddit'],
                                score=item['score'],
                                confidence=age_scores[likely_age] / max(1, sum(age_scores.values()))
                            )
                        )
                        break
        
        # Gender analysis
        gender_scores = {}
        for gender, keywords in self.patterns['gender_indicators'].items():
            score = sum(combined_text.count(keyword) for keyword in keywords)
            gender_scores[gender] = score
        
        if gender_scores and max(gender_scores.values()) > 0:
            likely_gender = max(gender_scores, key=gender_scores.get)
            demographics['likely_gender'] = likely_gender
            
            # Find citations for gender
            for content_type, item, content in all_content:
                for keyword in self.patterns['gender_indicators'][likely_gender]:
                    if keyword in content.lower():
                        citations['likely_gender'].append(
                            Citation(
                                content=content[:200] + "...",
                                url=item['url'],
                                timestamp=item['timestamp'],
                                type=content_type,
                                subreddit=item['subreddit'],
                                score=item['score'],
                                confidence=gender_scores[likely_gender] / max(1, sum(gender_scores.values()))
                            )
                        )
                        break
        
        # Location analysis (basic)
        location_patterns = [
            r'\b(?:from|in|live in|living in|located in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+,\s*[A-Z]{2})\b',  # City, State
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+area\b'
        ]
        
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            locations.extend(matches)
        
        if locations:
            location_count = Counter(locations)
            demographics['estimated_location'] = location_count.most_common(1)[0][0]
        
        return demographics, dict(citations)
    
    def analyze_personality(self, user_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[Citation]]]:
        """Analyze personality traits with citations"""
        personality = {}
        citations = defaultdict(list)
        
        all_text = []
        all_content = []
        
        # Combine all text content
        for post in user_data['posts']:
            content = f"{post['title']} {post['content']}"
            all_text.append(content.lower())
            all_content.append(('post', post, content))
        
        for comment in user_data['comments']:
            content = comment['content']
            all_text.append(content.lower())
            all_content.append(('comment', comment, content))
        
        combined_text = ' '.join(all_text)
        
        # Personality traits analysis
        trait_scores = {}
        for trait, keywords in self.patterns['personality_traits'].items():
            score = sum(combined_text.count(keyword) for keyword in keywords)
            trait_scores[trait] = score
        
        # Get top personality traits
        top_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        personality['personality_traits'] = [trait for trait, score in top_traits if score > 0]
        
        # Find citations for personality traits
        for trait, score in top_traits:
            if score > 0:
                for content_type, item, content in all_content:
                    for keyword in self.patterns['personality_traits'][trait]:
                        if keyword in content.lower():
                            citations['personality_traits'].append(
                                Citation(
                                    content=content[:200] + "...",
                                    url=item['url'],
                                    timestamp=item['timestamp'],
                                    type=content_type,
                                    subreddit=item['subreddit'],
                                    score=item['score'],
                                    confidence=score / max(1, sum(trait_scores.values()))
                                )
                            )
                            break
        
        return personality, dict(citations)
    
    def analyze_interests(self, user_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[Citation]]]:
        """Analyze user interests and expertise areas"""
        interests = {}
        citations = defaultdict(list)
        
        # Analyze subreddit activity
        subreddit_activity = defaultdict(int)
        subreddit_posts = defaultdict(list)
        
        for post in user_data['posts']:
            subreddit_activity[post['subreddit']] += 1
            subreddit_posts[post['subreddit']].append(post)
        
        for comment in user_data['comments']:
            subreddit_activity[comment['subreddit']] += 1
            subreddit_posts[comment['subreddit']].append(comment)
        
        # Top subreddits
        top_subreddits = sorted(subreddit_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        interests['favorite_subreddits'] = [sub for sub, count in top_subreddits]
        
        # Categorize interests based on subreddits
        interest_categories = {
            'technology': ['programming', 'technology', 'coding', 'javascript', 'python', 'webdev'],
            'gaming': ['gaming', 'games', 'wow', 'lol', 'minecraft', 'steam'],
            'finance': ['investing', 'stocks', 'personalfinance', 'cryptocurrency', 'bitcoin'],
            'fitness': ['fitness', 'bodybuilding', 'running', 'yoga', 'health'],
            'creative': ['art', 'design', 'photography', 'music', 'writing'],
            'lifestyle': ['food', 'cooking', 'travel', 'fashion', 'relationships']
        }
        
        primary_interests = []
        for category, keywords in interest_categories.items():
            score = sum(subreddit_activity.get(keyword, 0) for keyword in keywords)
            if score > 0:
                primary_interests.append((category, score))
        
        primary_interests.sort(key=lambda x: x[1], reverse=True)
        interests['primary_interests'] = [cat for cat, score in primary_interests[:5]]
        
        return interests, dict(citations)
    
    def analyze_sentiment(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze user's sentiment profile"""
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_content = 0
        
        all_content = []
        for post in user_data['posts']:
            all_content.append(f"{post['title']} {post['content']}")
        
        for comment in user_data['comments']:
            all_content.append(comment['content'])
        
        for content in all_content:
            if content.strip():
                blob = TextBlob(content)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment_scores['positive'] += 1
                elif polarity < -0.1:
                    sentiment_scores['negative'] += 1
                else:
                    sentiment_scores['neutral'] += 1
                
                total_content += 1
        
        if total_content > 0:
            for key in sentiment_scores:
                sentiment_scores[key] = sentiment_scores[key] / total_content
        
        return sentiment_scores
    
    def analyze_activity_patterns(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's activity patterns"""
        activity_by_hour = defaultdict(int)
        activity_by_day = defaultdict(int)
        
        all_timestamps = []
        for post in user_data['posts']:
            all_timestamps.append(post['timestamp'])
        
        for comment in user_data['comments']:
            all_timestamps.append(comment['timestamp'])
        
        for timestamp in all_timestamps:
            activity_by_hour[timestamp.hour] += 1
            activity_by_day[timestamp.strftime('%A')] += 1
        
        most_active_hour = max(activity_by_hour.items(), key=lambda x: x[1])[0] if activity_by_hour else 0
        most_active_day = max(activity_by_day.items(), key=lambda x: x[1])[0] if activity_by_day else 'Monday'
        
        return {
            'most_active_hour': most_active_hour,
            'most_active_day': most_active_day,
            'activity_by_hour': dict(activity_by_hour),
            'activity_by_day': dict(activity_by_day),
            'total_posts': len(user_data['posts']),
            'total_comments': len(user_data['comments']),
            'avg_posts_per_day': len(user_data['posts']) / max(1, (datetime.now() - user_data['account_created']).days)
        }
    
    def generate_persona(self, username: str, session_id: str) -> UserPersona:
        """Generate comprehensive user persona"""
        analysis_status[session_id] = {'status': 'extracting_data', 'progress': 10}
        
        # Extract user data
        user_data = self.extract_user_data(username)
        if not user_data:
            analysis_status[session_id] = {'status': 'error', 'message': 'User not found or data extraction failed'}
            return None
        
        analysis_status[session_id] = {'status': 'analyzing_demographics', 'progress': 30}
        
        # Analyze different aspects
        demographics, demo_citations = self.analyze_demographics(user_data)
        
        analysis_status[session_id] = {'status': 'analyzing_personality', 'progress': 50}
        personality, personality_citations = self.analyze_personality(user_data)
        
        analysis_status[session_id] = {'status': 'analyzing_interests', 'progress': 70}
        interests, interest_citations = self.analyze_interests(user_data)
        
        analysis_status[session_id] = {'status': 'analyzing_sentiment', 'progress': 85}
        sentiment_profile = self.analyze_sentiment(user_data)
        
        analysis_status[session_id] = {'status': 'analyzing_activity', 'progress': 95}
        activity_pattern = self.analyze_activity_patterns(user_data)
        
        # Combine all citations
        all_citations = {}
        all_citations.update(demo_citations)
        all_citations.update(personality_citations)
        all_citations.update(interest_citations)
        
        # Create persona
        persona = UserPersona(
            username=username,
            account_age=(datetime.now() - user_data['account_created']).days,
            total_karma=user_data['total_karma'],
            
            # Demographics
            estimated_age_range=demographics.get('estimated_age_range', 'Unknown'),
            likely_gender=demographics.get('likely_gender', 'Unknown'),
            estimated_location=demographics.get('estimated_location', 'Unknown'),
            occupation_indicators=[],
            
            # Personality
            personality_traits=personality.get('personality_traits', []),
            communication_style='Unknown',
            humor_style='Unknown',
            political_leaning='Unknown',
            
            # Interests
            primary_interests=interests.get('primary_interests', []),
            secondary_interests=[],
            expertise_areas=[],
            
            # Digital Behavior
            activity_pattern=activity_pattern,
            favorite_subreddits=interests.get('favorite_subreddits', []),
            engagement_style='Unknown',
            content_preference='Unknown',
            
            # Unique Insights
            sentiment_profile=sentiment_profile,
            linguistic_patterns={},
            behavioral_timeline=[],
            influence_network={},
            
            # Meta
            analysis_confidence=0.75,
            generation_timestamp=datetime.now(),
            citations=all_citations
        )
        
        analysis_status[session_id] = {'status': 'complete', 'progress': 100}
        return persona

# Initialize generator
generator = RedditPersonaGenerator(REDDIT_CONFIG)

def generate_persona_async(username: str, session_id: str):
    """Generate persona asynchronously"""
    try:
        persona = generator.generate_persona(username, session_id)
        if persona:
            analysis_results[session_id] = persona
    except Exception as e:
        analysis_status[session_id] = {'status': 'error', 'message': str(e)}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Start analysis"""
    username = request.json.get('username', '').strip()
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Start analysis in background thread
    thread = threading.Thread(target=generate_persona_async, args=(username, session_id))
    thread.start()
    
    return jsonify({'session_id': session_id})

@app.route('/status/<session_id>')
def status(session_id):
    """Get analysis status"""
    if session_id not in analysis_status:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(analysis_status[session_id])

@app.route('/result/<session_id>')
def result(session_id):
    """Get analysis result"""
    if session_id not in analysis_results:
        return jsonify({'error': 'Result not found'}), 404
    
    persona = analysis_results[session_id]
    
    # Convert persona to dictionary for JSON serialization
    persona_dict = asdict(persona)
    
    # Handle datetime serialization
    persona_dict['generation_timestamp'] = persona.generation_timestamp.isoformat()
    
    # Handle citations serialization
    for key, citations in persona_dict['citations'].items():
        for citation in citations:
            citation['timestamp'] = citation['timestamp'].isoformat()
    
    return jsonify(persona_dict)

@app.route('/chart/<session_id>/<chart_type>')
def chart(session_id, chart_type):
    """Generate charts"""
    if session_id not in analysis_results:
        return jsonify({'error': 'Result not found'}), 404
    
    persona = analysis_results[session_id]
    
    img = io.BytesIO()
    
    if chart_type == 'activity':
        # Activity chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Activity by hour
        if persona.activity_pattern.get('activity_by_hour'):
            hours = list(persona.activity_pattern['activity_by_hour'].keys())
            activity = list(persona.activity_pattern['activity_by_hour'].values())
            ax1.bar(hours, activity, color='#1f77b4')
            ax1.set_title('Activity by Hour')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Posts/Comments')
        
        # Activity by day
        if persona.activity_pattern.get('activity_by_day'):
            days = list(persona.activity_pattern['activity_by_day'].keys())
            day_activity = list(persona.activity_pattern['activity_by_day'].values())
            ax2.bar(days, day_activity, color='#ff7f0e')
            ax2.set_title('Activity by Day')
            ax2.set_xlabel('Day of Week')
            ax2.set_ylabel('Posts/Comments')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
    elif chart_type == 'sentiment':
        # Sentiment chart
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiments = list(persona.sentiment_profile.keys())
        values = list(persona.sentiment_profile.values())
        colors = ['#2ca02c', '#d62728', '#7f7f7f']
        
        ax.pie(values, labels=sentiments, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Sentiment Distribution')
        
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        plt.close()
    
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    Path('templates').mkdir(exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)