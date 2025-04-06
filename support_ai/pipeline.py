from typing import Dict, Any
from .analyzer import TicketAnalyzer

class SupportPipeline:
    def __init__(self):
        self.analyzer = TicketAnalyzer()
    
    def process(self, chat_text: str, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        # Analyze the ticket
        result = self.analyzer.analyze_ticket(chat_text, ticket_data)
        
        # Convert to dictionary format
        return {
            "summary": result.summary,
            "extracted_issue": result.issue,
            "suggested_solution": result.solution,
            "priority_level": result.priority,
            "assigned_team": result.team,
            "estimated_resolution_time": result.estimated_time,
            "confidence_score": result.confidence,
            "similar_cases": result.similar_cases,
            "action_items": result.action_items,
            "sentiment": result.sentiment
        }

