import pandas as pd
import subprocess
import os
from colorama import init, Fore, Style
from support_ai.data_loader import TicketDataLoader

# Initialize colorama
init()

def check_model_connection():
    print(f"\n{Fore.BLUE}Testing Mistral model connection...{Style.RESET_ALL}")
    try:
        # Simple test prompt
        result = subprocess.run(
            ['ollama', 'run', 'mistral', 'Say "test successful"'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"{Fore.GREEN}✓ Model connection successful{Style.RESET_ALL}")
            print(f"Response: {result.stdout.strip()}")
        else:
            print(f"{Fore.RED}✗ Model connection failed{Style.RESET_ALL}")
            print(f"Error: {result.stderr}")
            
    except FileNotFoundError:
        print(f"{Fore.RED}✗ Ollama not found. Is it installed?{Style.RESET_ALL}")
    except subprocess.TimeoutExpired:
        print(f"{Fore.RED}✗ Model connection timed out{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}✗ Unexpected error: {str(e)}{Style.RESET_ALL}")

def check_dataset():
    print(f"\n{Fore.BLUE}Checking dataset...{Style.RESET_ALL}")
    
    data_path = "[Usecase 7] AI-Driven Customer Support Enhancing Efficiency Through Multiagents/Historical_ticket_data.csv"
    
    if not os.path.exists(data_path):
        print(f"{Fore.RED}✗ Dataset file not found at: {data_path}{Style.RESET_ALL}")
        return
    
    try:
        # Load the dataset and print exact column names
        df = pd.read_csv(data_path)
        print("\nActual column names in CSV:")
        for col in df.columns:
            print(f"'{col}'")
        
        # Remove any leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Updated required columns matching the actual CSV
        required_columns = [
            "Issue Category",  # Check if it's actually "Issue_Category" or has spaces
            "Sentiment",
            "Priority",
            "Solution",
            "Resolution Status",  # Check if it's "Resolution_Status"
            "Date of Resolution"  # This might be used instead of "Resolution Time"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n{Fore.RED}✗ Missing required columns: {', '.join(missing_columns)}{Style.RESET_ALL}")
            return
            
        print(f"\n{Fore.GREEN}✓ Dataset loaded successfully{Style.RESET_ALL}")
        print(f"\nDataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"All columns: {', '.join(df.columns)}")
        
        # Show first few rows to verify data
        print("\nFirst few rows preview:")
        print(df.head(2).to_string())
        
    except Exception as e:
        print(f"{Fore.RED}✗ Error analyzing dataset: {str(e)}{Style.RESET_ALL}")

def main():
    print(f"{Fore.CYAN}=== AutoTriage.AI System Diagnostics ==={Style.RESET_ALL}")
    
    # Check model connection
    check_model_connection()
    
    # Check dataset
    check_dataset()

if __name__ == "__main__":
    main()
