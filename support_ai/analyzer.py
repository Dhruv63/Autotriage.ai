from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class AnalysisResult:
    summary: str
    issue: str
    solution: str
    priority: str
    team: str
    estimated_time: float
    confidence: float
    similar_cases: List[Dict]
    action_items: List[str]
    sentiment: str

class TicketAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.priority_levels = ["Low", "Medium", "High", "Critical"]
        self.teams = ["Technical", "Billing", "Product", "Security", "Customer Success"]
    
    def generate_summary(self, conversation: str) -> str:
        prompt = f"Summarize this customer support conversation in 1-2 sentences:\n\n{conversation}"
        return self.query_llm(prompt).strip()
    
    def extract_issue(self, conversation: str) -> str:
        prompt = f"Extract the main technical issue from this conversation:\n\n{conversation}"
        return self.query_llm(prompt).strip()
    
    def generate_solution(self, issue: str, similar_cases: List[Dict]) -> str:
        if similar_cases:
            # Use the solution from the most similar case
            return similar_cases[0]['solution']
        
        prompt = f"Suggest a solution for this technical issue:\n\n{issue}"
        return self.query_llm(prompt).strip()
    
    def determine_priority(self, issue: str, sentiment: str) -> str:
        keywords = {
            'Critical': ['urgent', 'emergency', 'critical', 'broken', 'down', 'error'],
            'High': ['important', 'serious', 'problem', 'failing'],
            'Medium': ['issue', 'bug', 'difficulty'],
            'Low': ['question', 'help', 'guidance']
        }
        
        issue_lower = issue.lower()
        
        # Check for urgent sentiment
        if sentiment.lower() in ['negative', 'urgent']:
            return 'High'
            
        # Check keywords
        for priority, words in keywords.items():
            if any(word in issue_lower for word in words):
                return priority
                
        return 'Medium'  # Default priority
    
    def determine_team(self, issue: str, priority: str) -> str:
        issue_lower = issue.lower()
        
        # Define team routing rules
        routing_rules = {
            'Technical': ['installation', 'error', 'bug', 'crash', 'technical'],
            'Billing': ['payment', 'charge', 'invoice', 'billing'],
            'Security': ['password', 'access', 'security', 'authentication'],
            'Product': ['feature', 'functionality', 'product'],
            'Customer Success': ['account', 'subscription', 'upgrade']
        }
        
        # Check each team's keywords
        for team, keywords in routing_rules.items():
            if any(keyword in issue_lower for keyword in keywords):
                return team
                
        return 'Technical'  # Default team
    
    def calculate_confidence(self, issue: str, similar_cases: List[Dict]) -> float:
        if not similar_cases:
            return 0.5  # Base confidence
            
        # Use the highest similarity score as confidence
        return max(case['similarity'] for case in similar_cases)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        # Vectorize the texts
        vectors = self.vectorizer.fit_transform([text1, text2])
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return similarity
    
    def query_llm(self, prompt: str) -> str:
        # Simplified LLM query - replace with actual implementation
        # This is a placeholder that returns basic responses
        if "summarize" in prompt.lower():
            return "Customer reported a technical issue and received assistance."
        elif "extract" in prompt.lower():
            return "Software installation failure with unknown error."
        elif "solution" in prompt.lower():
            return "Check system compatibility and retry installation."
        return "Response not available."
    
    def analyze_sentiment(self, text: str) -> str:
        prompt = f"Analyze the sentiment in this text and respond with one word (Positive/Negative/Neutral):\n{text}"
        return self.query_llm(prompt).strip().lower()
    
    def estimate_resolution_time(self, issue: str, priority: str, historical_data: Dict) -> float:
        similar_issues = [i for i, hist_issue in enumerate(historical_data['issues'])
                         if self.calculate_similarity(issue, hist_issue) > 0.7]
        
        # Default times based on priority
        priority_times = {
            "Critical": 4.0,
            "High": 8.0,
            "Medium": 24.0,
            "Low": 48.0
        }
        
        if similar_issues and 'resolution_times' in historical_data:
            times = [historical_data['resolution_times'][i] for i in similar_issues]
            return np.mean(times)
        
        return priority_times.get(priority, 24.0)
    
    def generate_action_items(self, issue: str, priority: str, team: str) -> List[str]:
        actions = [
            f"Route ticket to {team} team",
            f"Set priority as {priority}",
            "Send initial response to customer"
        ]
        
        if priority in ["Critical", "High"]:
            actions.append("Schedule immediate team review")
            actions.append("Prepare escalation path if needed")
        
        return actions
    
    def find_similar_cases(self, issue: str, historical_data: Dict) -> List[Dict]:
        similarities = [self.calculate_similarity(issue, hist_issue) 
                       for hist_issue in historical_data['issues']]
        
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        similar_cases = []
        for i in top_indices:
            if similarities[i] > 0.5:
                case = {
                    "issue": historical_data['issues'][i],
                    "solution": historical_data['solutions'][i],
                    "similarity": similarities[i],
                    "priority": historical_data.get('priorities', ['Medium'] * len(historical_data['issues']))[i],
                    "sentiment": historical_data.get('sentiments', ['Neutral'] * len(historical_data['issues']))[i],
                }
                # Add resolution time if available
                if 'resolution_times' in historical_data:
                    case["resolution_time"] = historical_data['resolution_times'][i]
                else:
                    case["resolution_time"] = 24.0  # Default to 24 hours
                
                similar_cases.append(case)
        
        return similar_cases
    
    def analyze_ticket(self, conversation: str, historical_data: Dict) -> AnalysisResult:
        # Extract key information
        summary = self.generate_summary(conversation)
        issue = self.extract_issue(conversation)
        sentiment = self.analyze_sentiment(conversation)
        
        # Determine routing and priority
        priority = self.determine_priority(issue, sentiment)
        team = self.determine_team(issue, priority)
        
        # Find similar cases and solution
        similar_cases = self.find_similar_cases(issue, historical_data)
        solution = self.generate_solution(issue, similar_cases)
        
        # Calculate confidence and estimated time
        confidence = self.calculate_confidence(issue, similar_cases)
        estimated_time = self.estimate_resolution_time(issue, priority, historical_data)
        
        # Generate action items
        action_items = self.generate_action_items(issue, priority, team)
        
        return AnalysisResult(
            summary=summary,
            issue=issue,
            solution=solution,
            priority=priority,
            team=team,
            estimated_time=estimated_time,
            confidence=confidence,
            similar_cases=similar_cases,
            action_items=action_items,
            sentiment=sentiment
        )


