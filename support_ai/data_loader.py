import pandas as pd
from typing import Dict, List
import logging

class TicketDataLoader:
    def __init__(self, file_path: str):
        self.logger = logging.getLogger(__name__)
        try:
            # Read the CSV file
            self.df = pd.read_csv(file_path)
            
            # Clean column names (remove whitespace and standardize)
            self.df.columns = self.df.columns.str.strip()
            
            # Map possible date column variations
            date_column_variants = [
                'Date of Resolution',
                'Resolution Date',
                'Date_of_Resolution',
                'resolution_date',
                'date_resolved'
            ]
            
            # Find the actual date column name
            date_column = None
            for variant in date_column_variants:
                if variant in self.df.columns:
                    date_column = variant
                    break
            
            # Convert date column if it exists
            if date_column:
                try:
                    self.df[date_column] = pd.to_datetime(self.df[date_column])
                except Exception as e:
                    self.logger.warning(f"Failed to parse date column: {str(e)}")
            else:
                self.logger.warning("No date column found. Adding default dates.")
                self.df['Date of Resolution'] = pd.Timestamp.now()
            
            # Ensure all required columns exist
            required_columns = {
                'Issue Category': 'Unknown Issue',
                'Sentiment': 'Neutral',
                'Priority': 'Medium',
                'Solution': 'No solution provided',
                'Resolution Status': 'Pending',
                'Resolution Time': 24.0  # Default 24 hours
            }
            
            # Add missing columns with default values
            for col, default_value in required_columns.items():
                if col not in self.df.columns:
                    self.logger.warning(f"Adding missing column '{col}' with default value")
                    self.df[col] = default_value
            
            # Clean text columns
            text_columns = ['Issue Category', 'Sentiment', 'Priority', 'Solution', 'Resolution Status']
            for col in text_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('Unknown').astype(str).str.strip()
            
            # Convert Resolution Time to float
            self.df['Resolution Time'] = pd.to_numeric(self.df['Resolution Time'], errors='coerce').fillna(24.0)
            
        except Exception as e:
            self.logger.error(f"Error initializing TicketDataLoader: {str(e)}")
            # Create empty DataFrame with required columns
            self.df = pd.DataFrame(columns=list(required_columns.keys()))
            self.df.loc[0] = list(required_columns.values())
    
    def get_training_data(self) -> Dict[str, List]:
        """Prepare historical data for the recommender system"""
        try:
            return {
                "issues": self.df["Issue Category"].tolist(),
                "sentiments": self.df["Sentiment"].tolist(),
                "priorities": self.df["Priority"].tolist(),
                "solutions": self.df["Solution"].tolist(),
                "resolution_times": self.df["Resolution Time"].tolist(),
                "statuses": self.df["Resolution Status"].tolist()
            }
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            # Return empty but valid data structure
            return {
                "issues": [], 
                "sentiments": [], 
                "priorities": [],
                "solutions": [], 
                "resolution_times": [], 
                "statuses": []
            }
    
    def get_similar_cases(self, issue_category: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get similar historical cases for a given issue category"""
        similar_cases = self.df[self.df["Issue Category"] == issue_category].head(limit)
        return similar_cases.to_dict('records')

<<<<<<< HEAD

=======
>>>>>>> 9aea4644744df0c5ff9b7bfec7c7e29a98148e7c
