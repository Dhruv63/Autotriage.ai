from support_ai.pipeline import SupportPipeline
from support_ai.data_loader import TicketDataLoader

def process_support_ticket(conversation_text: str):
    # 1. Initialize the data loader with historical data
    data_loader = TicketDataLoader("[Usecase 7] AI-Driven Customer Support Enhancing Efficiency Through Multiagents/Historical_ticket_data.csv")
    historical_data = data_loader.get_training_data()

    # 2. Create pipeline instance
    pipeline = SupportPipeline()

    # 3. Process the conversation
    result = pipeline.process(
        chat_text=conversation_text,
        ticket_data=historical_data
    )

    # 4. Print results
    print("\n=== AutoTriage.AI Analysis ===")
    print("Let AI diagnose your support mess")
    print("-" * 40)
    print(f"Summary: {result['summary']}")
    print(f"Extracted Issue: {result['extracted_issue']}")
    print(f"Suggested Solution: {result['suggested_solution']}")
    print(f"Confidence Score: {result['confidence_score']:.2f}")

    return result

# Example usage
if __name__ == "__main__":
    # Example conversation
    conversation = """
    Customer: Hi there! I've been trying to install your software but it keeps failing at 75% with an unknown error.
    Agent: Hello! Could you share more details about the error message?
    Customer: It just says "Installation failed" and nothing else. I've tried three times already.
    """
    
    result = process_support_ticket(conversation)
