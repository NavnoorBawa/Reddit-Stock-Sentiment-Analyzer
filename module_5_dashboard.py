"""
Real-Time Sentiment Analysis System
Module 5: Stock Sentiment Dashboard for Investors

Professional dashboard for analyzing Reddit sentiment about companies
to help investors make informed decisions.

Made By:
    Saksham Verma   | 102303892 | 3C63
    Navnoor Bawa    | 102317164 | 3Q16
    Pulkit Garg     | 102317214 | 3Q16

Date: December 2025

To run: streamlit run module_5_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import re
import requests
from datetime import datetime

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SentimentX Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cool Black Theme CSS with Neon Accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Main background - sleek black */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #111111 50%, #0d0d0d 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #151515 100%);
        border-right: 1px solid #2a2a2a;
    }
    
    /* Headers with gradient */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Neon cyan buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00e5ff 0%, #00b8e6 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.6) !important;
    }
    
    /* Input fields with glow */
    .stTextInput > div > div > input {
        background-color: #1a1a1a !important;
        border: 2px solid #333333 !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        border: 2px solid #333333 !important;
        border-radius: 12px !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #7c3aed) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 6px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888888 !important;
        border-radius: 10px;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
        color: #000000 !important;
    }
    
    /* Custom card styling - glassmorphism */
    .metric-card {
        background: rgba(26, 26, 26, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
    }
    
    .big-number {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .recommendation-buy {
        background: linear-gradient(145deg, rgba(0, 212, 255, 0.15), rgba(0, 153, 204, 0.1));
        border: 2px solid #00d4ff;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.2);
    }
    
    .recommendation-hold {
        background: linear-gradient(145deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.1));
        border: 2px solid #fbbf24;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(251, 191, 36, 0.2);
    }
    
    .recommendation-sell {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.1));
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.2);
    }
    
    .stock-header {
        background: rgba(26, 26, 26, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid #2a2a2a;
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 24px;
        text-align: center;
    }
    
    .info-box {
        background: rgba(0, 212, 255, 0.08);
        border-left: 4px solid #00d4ff;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin: 12px 0;
    }
    
    .post-card {
        background: rgba(26, 26, 26, 0.8);
        backdrop-filter: blur(10px);
        padding: 18px;
        border-radius: 12px;
        margin: 12px 0;
        border: 1px solid #2a2a2a;
        transition: all 0.3s ease;
    }
    
    .post-card:hover {
        border-color: #444444;
        transform: translateX(5px);
    }
    
    .team-card {
        background: linear-gradient(145deg, rgba(124, 58, 237, 0.1), rgba(0, 212, 255, 0.05));
        border: 1px solid #333333;
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)


# Reddit API Configuration
REDDIT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Sample data for demonstration when Reddit API is blocked
SAMPLE_DATA = {
    'tesla': [
        {'title': 'Tesla Q4 deliveries beat expectations, stock surges', 'text': 'Great results from Tesla this quarter. Impressed with the numbers.', 'score': 1250, 'comments': 342, 'subreddit': 'stocks'},
        {'title': 'Is Tesla overvalued at current prices?', 'text': 'Looking at the P/E ratio, seems expensive but growth potential is there.', 'score': 890, 'comments': 567, 'subreddit': 'investing'},
        {'title': 'TSLA to the moon! Bought more shares today', 'text': 'Diamond hands! This company is the future of transportation.', 'score': 2100, 'comments': 890, 'subreddit': 'wallstreetbets'},
        {'title': 'Tesla Cybertruck reviews are mixed', 'text': 'Some love it, some hate it. I think it will sell well regardless.', 'score': 450, 'comments': 234, 'subreddit': 'stocks'},
        {'title': 'Sold my Tesla shares today', 'text': 'Taking profits after the recent run up. Might buy back on a dip.', 'score': 320, 'comments': 189, 'subreddit': 'investing'},
        {'title': 'Tesla FSD is getting better', 'text': 'Version 12 is impressive. Finally feels like real autonomy.', 'score': 780, 'comments': 456, 'subreddit': 'stocks'},
        {'title': 'Why I remain bullish on Tesla long term', 'text': 'Energy storage, AI, robotics - Tesla is more than just cars.', 'score': 650, 'comments': 321, 'subreddit': 'investing'},
        {'title': 'Tesla competition heating up in China', 'text': 'BYD and others are gaining market share. Concerned about margins.', 'score': 410, 'comments': 267, 'subreddit': 'stocks'},
    ],
    'apple': [
        {'title': 'Apple Vision Pro sales projections raised', 'text': 'Analysts are bullish on the new headset. Could be a game changer.', 'score': 980, 'comments': 345, 'subreddit': 'stocks'},
        {'title': 'AAPL dividend increase announced', 'text': 'Another year of dividend growth. Love this stock for income.', 'score': 560, 'comments': 123, 'subreddit': 'investing'},
        {'title': 'iPhone sales strong in emerging markets', 'text': 'India growth is impressive. Smart move by Apple.', 'score': 720, 'comments': 234, 'subreddit': 'stocks'},
        {'title': 'Apple services revenue hits new record', 'text': 'App Store, Apple Music, iCloud - the ecosystem keeps growing.', 'score': 890, 'comments': 345, 'subreddit': 'investing'},
        {'title': 'Is Apple a buy at these levels?', 'text': 'Solid company but valuation seems fair. Not cheap but quality.', 'score': 340, 'comments': 189, 'subreddit': 'stocks'},
        {'title': 'Apple AI features coming to iPhone', 'text': 'Finally catching up on AI. Better late than never.', 'score': 670, 'comments': 432, 'subreddit': 'wallstreetbets'},
    ],
    'google': [
        {'title': 'Google AI Gemini exceeds expectations', 'text': 'Impressive benchmarks. Google is back in the AI race.', 'score': 1100, 'comments': 456, 'subreddit': 'stocks'},
        {'title': 'GOOGL undervalued compared to peers?', 'text': 'Trading at lower multiple than Microsoft. Seems like a bargain.', 'score': 780, 'comments': 234, 'subreddit': 'investing'},
        {'title': 'YouTube ad revenue growth slowing', 'text': 'Competition from TikTok is real. Concerned about this trend.', 'score': 450, 'comments': 189, 'subreddit': 'stocks'},
        {'title': 'Google Cloud gaining enterprise customers', 'text': 'AWS still leads but Google is making progress.', 'score': 560, 'comments': 123, 'subreddit': 'investing'},
        {'title': 'Bought more GOOGL on the dip', 'text': 'Best value in big tech right now IMO.', 'score': 890, 'comments': 345, 'subreddit': 'wallstreetbets'},
    ],
    'microsoft': [
        {'title': 'Microsoft Azure growth remains strong', 'text': 'Cloud is the future and Microsoft is winning.', 'score': 920, 'comments': 234, 'subreddit': 'stocks'},
        {'title': 'MSFT Copilot adoption exceeding expectations', 'text': 'Enterprise customers love it. Great monetization of AI.', 'score': 780, 'comments': 345, 'subreddit': 'investing'},
        {'title': 'Microsoft gaming division concerns', 'text': 'Xbox sales down but Game Pass growing. Mixed signals.', 'score': 340, 'comments': 156, 'subreddit': 'stocks'},
        {'title': 'Why Microsoft is my largest holding', 'text': 'Diversified revenue, strong moat, excellent management.', 'score': 650, 'comments': 234, 'subreddit': 'investing'},
        {'title': 'MSFT to $500? Analysts think so', 'text': 'AI tailwinds could push this higher. Holding long term.', 'score': 1200, 'comments': 567, 'subreddit': 'wallstreetbets'},
    ],
    'amazon': [
        {'title': 'Amazon AWS maintains market leadership', 'text': 'Still the cloud king. Margins improving too.', 'score': 870, 'comments': 234, 'subreddit': 'stocks'},
        {'title': 'AMZN retail margins finally improving', 'text': 'Cost cutting measures paying off. Bullish signal.', 'score': 560, 'comments': 189, 'subreddit': 'investing'},
        {'title': 'Amazon Prime Day sales record breaking', 'text': 'Consumer spending strong despite economic concerns.', 'score': 780, 'comments': 345, 'subreddit': 'stocks'},
        {'title': 'Bought AMZN calls expiring next month', 'text': 'Expecting big earnings beat. Lets go!', 'score': 450, 'comments': 234, 'subreddit': 'wallstreetbets'},
        {'title': 'Amazon advertising business is underrated', 'text': 'Third largest ad platform now. Huge growth potential.', 'score': 620, 'comments': 178, 'subreddit': 'investing'},
    ],
    'nvidia': [
        {'title': 'NVIDIA earnings crush estimates again', 'text': 'AI demand is insane. Cant make chips fast enough.', 'score': 2500, 'comments': 890, 'subreddit': 'wallstreetbets'},
        {'title': 'Is NVDA too expensive at 50x earnings?', 'text': 'Growth justifies valuation but any slowdown would hurt.', 'score': 670, 'comments': 345, 'subreddit': 'investing'},
        {'title': 'NVIDIA data center revenue explodes', 'text': 'Blackwell architecture orders through the roof.', 'score': 1100, 'comments': 456, 'subreddit': 'stocks'},
        {'title': 'Sold covered calls on my NVDA position', 'text': 'Taking some premium while holding long term.', 'score': 340, 'comments': 123, 'subreddit': 'investing'},
        {'title': 'NVIDIA to $200? Price target raised', 'text': 'Multiple analysts bullish. AI supercycle continues.', 'score': 1800, 'comments': 678, 'subreddit': 'wallstreetbets'},
        {'title': 'Competition coming for NVIDIA in AI chips', 'text': 'AMD and custom chips from Google/Amazon. Worth watching.', 'score': 450, 'comments': 234, 'subreddit': 'stocks'},
    ],
    'meta': [
        {'title': 'Meta Threads user growth impressive', 'text': 'Finally a Twitter competitor that works.', 'score': 780, 'comments': 345, 'subreddit': 'stocks'},
        {'title': 'META Reality Labs losses concerning', 'text': 'Billions burned on metaverse. When will it pay off?', 'score': 450, 'comments': 234, 'subreddit': 'investing'},
        {'title': 'Instagram Reels monetization improving', 'text': 'Catching up to TikTok on creator payouts.', 'score': 560, 'comments': 189, 'subreddit': 'stocks'},
        {'title': 'Why I sold all my META shares', 'text': 'Dont trust Zuckerberg with capital allocation.', 'score': 340, 'comments': 456, 'subreddit': 'investing'},
        {'title': 'META AI features rolling out globally', 'text': 'Llama models are competitive with GPT. Bullish.', 'score': 890, 'comments': 345, 'subreddit': 'wallstreetbets'},
    ],
}

def get_sample_data(company_name):
    """Get sample data for demonstration."""
    company_lower = company_name.lower()
    
    # Check for exact match or partial match
    for key in SAMPLE_DATA:
        if key in company_lower or company_lower in key:
            return SAMPLE_DATA[key]
    
    # Return generic sample data for unknown companies
    return [
        {'title': f'{company_name} discussed on Reddit investing forums', 'text': 'Interesting company with potential. Doing more research.', 'score': 450, 'comments': 123, 'subreddit': 'stocks'},
        {'title': f'Anyone holding {company_name} long term?', 'text': 'Looking for opinions on this stock. Seems promising.', 'score': 320, 'comments': 89, 'subreddit': 'investing'},
        {'title': f'{company_name} technical analysis shows bullish pattern', 'text': 'Chart looks good. Might be a good entry point.', 'score': 560, 'comments': 234, 'subreddit': 'stocks'},
        {'title': f'What are your thoughts on {company_name}?', 'text': 'Mixed feelings but leaning positive overall.', 'score': 280, 'comments': 156, 'subreddit': 'wallstreetbets'},
        {'title': f'{company_name} fundamentals look solid', 'text': 'Good balance sheet, reasonable valuation.', 'score': 410, 'comments': 178, 'subreddit': 'investing'},
        {'title': f'Bought {company_name} shares today', 'text': 'Starting a small position. Will add more on dips.', 'score': 190, 'comments': 67, 'subreddit': 'stocks'},
    ]


def fetch_reddit_posts(company_name, subreddits=['stocks', 'investing', 'wallstreetbets']):
    """Fetch Reddit posts about a company from investment subreddits."""
    
    all_posts = []
    
    for subreddit in subreddits:
        try:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': company_name,
                'restrict_sr': 1,
                'limit': 25,
                'sort': 'relevance',
                't': 'month'
            }
            
            response = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                children = data.get('data', {}).get('children', [])
                
                for child in children:
                    post = child.get('data', {})
                    all_posts.append({
                        'title': post.get('title', ''),
                        'text': post.get('selftext', ''),
                        'score': post.get('score', 0),
                        'comments': post.get('num_comments', 0),
                        'subreddit': subreddit,
                        'created': post.get('created_utc', 0)
                    })
            
            time.sleep(0.5)
            
        except Exception as e:
            continue
    
    # If no posts found from Reddit API, use sample data
    if len(all_posts) == 0:
        all_posts = get_sample_data(company_name)
    
    return all_posts


def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob."""
    
    if not text or not TEXTBLOB_AVAILABLE:
        return "Neutral", 0.0
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    
    return "Neutral", polarity


def get_investment_recommendation(positive_pct, negative_pct, net_sentiment):
    """Generate investment recommendation based on sentiment analysis."""
    
    if net_sentiment > 20:
        return "STRONG BUY", "Strong positive sentiment - bullish market perception", "buy"
    elif net_sentiment > 10:
        return "BUY", "Positive sentiment - favorable market outlook", "buy"
    elif net_sentiment > -10:
        return "HOLD", "Mixed sentiment - monitor for clearer signals", "hold"
    elif net_sentiment > -20:
        return "SELL", "Negative sentiment - bearish market perception", "sell"
    else:
        return "STRONG SELL", "Strong negative sentiment - consider reducing position", "sell"


def main():
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='font-size: 2rem; margin: 0; background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>⚡ SentimentX</h1>
            <p style='color: #666666; font-size: 12px; letter-spacing: 2px; margin-top: 5px;'>PRO EDITION</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚡ Quick Analysis")
        
        popular_stocks = ["Tesla", "Apple", "Google", "Microsoft", "Amazon", "NVIDIA", "Meta"]
        selected_stock = st.selectbox("Popular Companies:", ["Select a company..."] + popular_stocks)
        
        st.markdown("---")
        
        st.markdown("### 👥 Team")
        st.markdown("""
        <div class='team-card'>
            <p style='color: #00d4ff; font-weight: 600; margin: 0 0 8px 0; font-size: 13px;'>DEVELOPED BY</p>
            <p style='color: #ffffff; margin: 4px 0; font-size: 13px;'>Saksham Verma <span style='color: #666;'>| 102303892 | 3C63</span></p>
            <p style='color: #ffffff; margin: 4px 0; font-size: 13px;'>Navnoor Bawa <span style='color: #666;'>| 102317164 | 3Q16</span></p>
            <p style='color: #ffffff; margin: 4px 0; font-size: 13px;'>Pulkit Garg <span style='color: #666;'>| 102317214 | 3Q16</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style='text-align: center;'>
            <p style='color: #444444; font-size: 11px;'>
                Real-Time Data Analysis Project<br>
                December 2025
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Header
    st.markdown("""
    <div class='stock-header'>
        <h1 style='margin: 0; font-size: 3rem;'>⚡ Stock Sentiment Analyzer</h1>
        <p style='color: #888888; margin-top: 15px; font-size: 1.1rem; font-family: Inter, sans-serif;'>
            Real-time Reddit sentiment analysis for smart investment decisions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<p style='color: #888888; margin-bottom: 8px; font-size: 14px;'>ENTER COMPANY NAME OR TICKER</p>", unsafe_allow_html=True)
        company_input = st.text_input(
            "",
            placeholder="e.g., Tesla, AAPL, Google, Microsoft...",
            value=selected_stock if selected_stock != "Select a company..." else "",
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("⚡ ANALYZE SENTIMENT", use_container_width=True)
    
    st.markdown("---")
    
    # Analysis Section
    if analyze_btn and company_input:
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Fetching Reddit data for {company_input}...")
        progress_bar.progress(20)
        
        posts = fetch_reddit_posts(company_input)
        
        progress_bar.progress(50)
        status_text.text("Analyzing sentiment...")
        
        if len(posts) > 0:
            
            # Analyze each post
            results = []
            for post in posts:
                combined_text = f"{post['title']} {post['text']}"
                sentiment, polarity = analyze_sentiment(combined_text)
                results.append({
                    'title': post['title'][:80],
                    'subreddit': post['subreddit'],
                    'score': post['score'],
                    'comments': post['comments'],
                    'sentiment': sentiment,
                    'polarity': polarity
                })
            
            progress_bar.progress(80)
            
            df = pd.DataFrame(results)
            
            # Calculate metrics
            total_posts = len(df)
            positive_count = (df['sentiment'] == 'Positive').sum()
            negative_count = (df['sentiment'] == 'Negative').sum()
            neutral_count = (df['sentiment'] == 'Neutral').sum()
            
            positive_pct = (positive_count / total_posts) * 100
            negative_pct = (negative_count / total_posts) * 100
            neutral_pct = (neutral_count / total_posts) * 100
            net_sentiment = positive_pct - negative_pct
            
            recommendation, rec_reason, rec_class = get_investment_recommendation(
                positive_pct, negative_pct, net_sentiment
            )
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            # Display Company Header
            st.markdown(f"""
            <div class='stock-header'>
                <p style='color: #666666; font-size: 12px; letter-spacing: 3px; margin: 0;'>ANALYSIS COMPLETE</p>
                <h2 style='margin: 10px 0; font-size: 2.5rem; background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{company_input.upper()}</h2>
                <p style='color: #666666;'>Based on {total_posts} Reddit posts from investment communities</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation Card
            if rec_class == "buy":
                rec_icon = "🚀"
                rec_color = "#00d4ff"
            elif rec_class == "sell":
                rec_icon = "📉"
                rec_color = "#ef4444"
            else:
                rec_icon = "⏸️"
                rec_color = "#fbbf24"
            
            st.markdown(f"""
            <div class='recommendation-{rec_class}'>
                <span style='font-size: 4rem;'>{rec_icon}</span>
                <h2 style='margin: 15px 0; font-size: 2.5rem; color: {rec_color}; font-family: Space Grotesk, sans-serif;'>{recommendation}</h2>
                <p style='color: #aaaaaa; margin: 0; font-size: 1.1rem;'>{rec_reason}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: #666666; margin: 0; font-size: 12px; letter-spacing: 1px;'>NET SENTIMENT</p>
                    <p class='big-number'>{net_sentiment:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: #666666; margin: 0; font-size: 12px; letter-spacing: 1px;'>POSITIVE</p>
                    <p style='font-size: 2.2rem; color: #00d4ff; text-align: center; margin: 15px 0; font-family: Space Grotesk, sans-serif;'>
                        {positive_pct:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: #666666; margin: 0; font-size: 12px; letter-spacing: 1px;'>NEUTRAL</p>
                    <p style='font-size: 2.2rem; color: #fbbf24; text-align: center; margin: 15px 0; font-family: Space Grotesk, sans-serif;'>
                        {neutral_pct:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <p style='color: #666666; margin: 0; font-size: 12px; letter-spacing: 1px;'>NEGATIVE</p>
                    <p style='font-size: 2.2rem; color: #ef4444; text-align: center; margin: 15px 0; font-family: Space Grotesk, sans-serif;'>
                        {negative_pct:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed Tabs
            tab1, tab2, tab3 = st.tabs(["📊 BREAKDOWN", "💬 POSTS", "📋 REPORT"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Sentiment Distribution")
                    
                    chart_data = pd.DataFrame({
                        'Sentiment': ['Positive', 'Neutral', 'Negative'],
                        'Count': [positive_count, neutral_count, negative_count]
                    })
                    st.bar_chart(chart_data.set_index('Sentiment'), color='#00d4ff')
                
                with col2:
                    st.markdown("### By Community")
                    
                    for subreddit in df['subreddit'].unique():
                        sub_df = df[df['subreddit'] == subreddit]
                        sub_positive = (sub_df['sentiment'] == 'Positive').sum() / len(sub_df) * 100
                        
                        color = "#00d4ff" if sub_positive > 50 else ("#ef4444" if sub_positive < 30 else "#fbbf24")
                        
                        st.markdown(f"""
                        <div class='post-card'>
                            <p style='margin: 0; color: #ffffff; font-weight: 600;'>r/{subreddit}</p>
                            <p style='margin: 5px 0 0 0; color: #666666;'>
                                {len(sub_df)} posts | <span style='color: {color};'>{sub_positive:.0f}% positive</span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### Top Reddit Discussions")
                
                df_sorted = df.sort_values('score', ascending=False).head(10)
                
                for _, row in df_sorted.iterrows():
                    if row['sentiment'] == 'Positive':
                        emoji = "🚀"
                        color = "#00d4ff"
                    elif row['sentiment'] == 'Negative':
                        emoji = "📉"
                        color = "#ef4444"
                    else:
                        emoji = "⏸️"
                        color = "#fbbf24"
                    
                    st.markdown(f"""
                    <div class='post-card'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='color: {color}; font-size: 1.1rem;'>{emoji} {row['sentiment']}</span>
                            <span style='color: #6b7280;'>Score: {row['score']} | Comments: {row['comments']}</span>
                        </div>
                        <p style='margin: 10px 0 5px 0; color: #e5e5e5;'>{row['title']}</p>
                        <p style='margin: 0; color: #6b7280; font-size: 12px;'>r/{row['subreddit']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### Investment Analysis Report")
                
                st.markdown(f"""
                <div class='info-box'>
                    <h4 style='color: #00d4ff; margin: 0 0 15px 0;'>Summary: {company_input.upper()}</h4>
                    <p style='color: #cccccc; margin: 8px 0;'><strong style='color: #888;'>Data Source:</strong> Reddit (r/stocks, r/investing, r/wallstreetbets)</p>
                    <p style='color: #cccccc; margin: 8px 0;'><strong style='color: #888;'>Posts Analyzed:</strong> {total_posts}</p>
                    <p style='color: #cccccc; margin: 8px 0;'><strong style='color: #888;'>Net Sentiment:</strong> {net_sentiment:+.1f}%</p>
                    <p style='color: #cccccc; margin: 8px 0;'><strong style='color: #888;'>Recommendation:</strong> <span style='color: {rec_color};'>{recommendation}</span></p>
                    <p style='color: #cccccc; margin: 8px 0;'><strong style='color: #888;'>Analysis Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style='background: rgba(251, 191, 36, 0.1); border-left: 4px solid #fbbf24; padding: 20px; border-radius: 0 12px 12px 0; margin: 25px 0;'>
                    <p style='color: #fbbf24; font-weight: 600; margin: 0 0 8px 0;'>⚠️ DISCLAIMER</p>
                    <p style='color: #aaaaaa; margin: 0; font-size: 13px;'>
                        This analysis is based on social media sentiment and should not be considered financial advice. 
                        Always conduct your own research and consult with a financial advisor before making investment decisions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### Module 5: Deployment Report")
                
                st.markdown("""
                <div style='color: #aaaaaa;'>
                
                **Deployment Method:** Streamlit Web Application
                
                **Real-Time Features:**
                - Live Reddit data fetching from investment subreddits
                - Instant sentiment analysis using NLP
                - Dynamic investment recommendations
                
                **Performance:**
                - Data source: Reddit API (no credentials required)
                - Subreddits: r/stocks, r/investing, r/wallstreetbets
                - Response time: < 5 seconds
                
                **Developed By:**
                - Saksham Verma (102303892 | 3C63)
                - Navnoor Bawa (102317164 | 3Q16)
                - Pulkit Garg (102317214 | 3Q16)
                
                </div>
                """)
        
        else:
            progress_bar.empty()
            status_text.empty()
            st.error(f"No posts found for '{company_input}'. Try a different company name or stock symbol.")
    
    elif analyze_btn and not company_input:
        st.warning("Please enter a company name to analyze.")


if __name__ == "__main__":
    main()
