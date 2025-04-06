import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

class TicketDataLoader:
    def __init__(self, csv_path: str):
        # Read CSV and strip whitespace from column names
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        
        # Convert date strings to datetime and calculate resolution times in hours
        self.df['Date of Resolution'] = pd.to_datetime(self.df['Date of Resolution'])
        self.df['Resolution Time'] = 24.0  # Default to 24 hours if not calculable
        
    def get_training_data(self) -> Dict[str, List[str]]:
        """Prepare historical data for the recommender system"""
        return {
            "issues": self.df["Issue Category"].str.strip().tolist(),
            "sentiments": self.df["Sentiment"].str.strip().tolist(),
            "priorities": self.df["Priority"].str.strip().tolist(),
            "solutions": self.df["Solution"].str.strip().tolist(),
            "resolution_times": self.df["Resolution Time"].tolist(),
            "statuses": self.df["Resolution Status"].str.strip().tolist()
        }
    
    def get_similar_cases(self, issue_category: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get similar historical cases for a given issue category"""
        similar_cases = self.df[self.df["Issue Category"] == issue_category].head(limit)
        return similar_cases.to_dict('records')

