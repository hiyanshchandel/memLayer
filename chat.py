from get_response import get_chatbot_response, get_history
from upsert_to_qdrant import upsert_vector_batch    
def chat():
    """Main chat loop"""
    print("Chatbot started! Type 'quit' to exit.\n")
    message_count = 0  # Track total messages (user + AI)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        if not user_input.strip():
            continue
        
        # Increment for user message
        message_count += 1
        
        # Get response
        response = get_chatbot_response(user_input)
        print(f"Chatbot: {response}\n")
        
        # Increment for AI message
        message_count += 1
        
        if message_count % 4 == 0:
            history = get_history()
            upsert_history = upsert_vector_batch('memLayer',history)

if __name__ == "__main__":
    chat()