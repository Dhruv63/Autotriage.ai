import streamlit as st
import pandas as pd
from support_ai.pipeline import SupportPipeline
from support_ai.data_loader import TicketDataLoader
import plotly.graph_objects as go
import time
import os
import logging

# Page config
st.set_page_config(
    page_title="AutoTriage.AI",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

def load_data():
    try:
        data_path = "[Usecase 7] AI-Driven Customer Support Enhancing Efficiency Through Multiagents/Historical_ticket_data.csv"
        if not os.path.exists(data_path):
            st.error(f"Dataset not found at: {data_path}")
            st.info("Please ensure the dataset file is in the correct location.")
            return None
        
        loader = TicketDataLoader(data_path)
        if loader.df.empty:
            st.warning("No data loaded. Using default values for demonstration.")
        return loader
    except Exception as e:
        st.error("Error loading data. Please check the logs for details.")
        logging.error(f"Data loading error: {str(e)}")
        return None

def analyze_conversation(conversation_text, historical_data):
    pipeline = SupportPipeline()
    return pipeline.process(
        chat_text=conversation_text,
        ticket_data=historical_data
    )

def display_metrics(result):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence Score", f"{result['confidence_score']:.2%}")
    with col2:
        st.metric("Similar Cases Found", len(result.get('similar_cases', [])))
    with col3:
        # Assuming we track processing time
        st.metric("Processing Time", "< 1 sec")

def main():
    # Sidebar
    st.sidebar.title("🔍 AutoTriage.AI")
    st.sidebar.info(
        "Let AI diagnose your support mess.\n\n"
        "Our AI-powered system automatically analyzes, categorizes, "
        "and provides solutions for your support tickets."
    )
    
    # Main content
    st.title("AutoTriage.AI")
    st.subheader("Let AI diagnose your support mess")
    
    # Load data
    data_loader = load_data()
    if data_loader is None:
        st.stop()
    
    historical_data = data_loader.get_training_data()
    
    # Input section
    st.subheader("📝 Conversation Input")
    input_method = st.radio(
        "Choose input method:",
        ["Enter Text", "Upload File", "Sample Conversations"]
    )
    
    conversation_text = ""
    
    if input_method == "Enter Text":
        conversation_text = st.text_area(
            "Enter the conversation:",
            height=200,
            placeholder="Paste your customer support conversation here..."
        )
        
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload conversation file", type=['txt'])
        if uploaded_file:
            conversation_text = uploaded_file.getvalue().decode()
            
    else:  # Sample Conversations
        samples = {
            "Software Installation Issue": """Customer: "Hi there! I've been trying to install the latest update for your design software for hours. It keeps failing at 75% with an 'unknown error.' What's wrong?"
Agent: "Hello! Thank you for reaching out. Let me help troubleshoot. Could you share a screenshot of the error message and confirm your operating system version?"
Customer: "Sure, it's Windows 11. Here's the screenshot: [image link]. I've restarted twice, same issue."
Agent: "Thank you for the details. This is a known conflict with third-party antivirus tools. Could you temporarily disable your antivirus and retry? I'll also send a direct download link as a workaround."
Customer: "Oh, disabling the antivirus worked! Installation completed. Thanks for your help!"
""",
            "Payment Gateway Integration": """Customer: "Hi, this is urgent! Your API is rejecting our payment gateway integration. Error: 'Invalid SSL certificate.' Our cert is valid and up-to-date!"
Agent: "Hello! Let's investigate immediately. Could you share the output from openssl s_client -connect yourgateway.com:443?"
Customer: "Here's the terminal output: [text]. See? No errors here."
Agent: "Thank you! Our system requires TLS 1.3, but your server supports only up to TLS 1.2. Upgrading the protocol will resolve the authentication error."
Customer: "Upgrading worked! Thanks for the quick fix!"
"""
        }
        selected_sample = st.selectbox("Choose a sample conversation:", list(samples.keys()))
        conversation_text = samples[selected_sample]
        st.text_area("Sample conversation:", conversation_text, height=200, disabled=True)
    
    # Analysis button
    if st.button("🔍 Analyze Conversation") and conversation_text:
        with st.spinner("Analyzing conversation..."):
            # Add progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Process the conversation
            result = analyze_conversation(conversation_text, historical_data)
            
            # Display results
            st.success("Analysis completed!")
            
            # Metrics
            display_metrics(result)
            
            # Results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔍 Similar Cases", "📈 Insights"])
            
            with tab1:
                st.subheader("Analysis Results")
                st.write("**Summary:**", result['summary'])
                st.write("**Extracted Issue:**", result['extracted_issue'])
                st.write("**Suggested Solution:**", result['suggested_solution'])
                
                # Updated confidence visualization using go.Figure
                confidence = result['confidence_score']
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Solution Confidence Score"},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "red"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': confidence
                        }
                    }
                ))
                
                # Update layout for better visualization
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=50, b=10, pad=8)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Similar Historical Cases")
                if 'similar_cases' in result:
                    for i, case in enumerate(result['similar_cases'], 1):
                        with st.expander(f"Case {i}: {case['issue'][:50]}..."):
                            st.write("**Priority:**", case.get('priority', 'Not specified'))
                            st.write("**Sentiment:**", case.get('sentiment', 'Not specified'))
                            st.write("**Solution:**", case.get('solution', 'No solution available'))
                            if 'similarity' in case:
                                st.write("**Similarity Score:**", f"{case['similarity']:.2f}")
                            if 'resolution_time' in case:
                                st.write("**Resolution Time:**", f"{case['resolution_time']:.1f} hours")
            
            with tab3:
                st.subheader("Conversation Insights")
                # Add any additional insights, charts, or analytics here
                
            # Add to history
            st.session_state.history.append({
                'timestamp': pd.Timestamp.now(),
                'summary': result['summary'],
                'confidence': result['confidence_score']
            })
    
    # History section
    if st.session_state.history:
        st.subheader("📚 Analysis History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

if __name__ == "__main__":
    main()




