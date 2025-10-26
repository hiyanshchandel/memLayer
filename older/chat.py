from older.get_response import get_chatbot_response, get_history
from older.upsert_to_qdrant import upsert_vector, get_insights
from older.deduplication import deduplication_decision, get_similar, merge_memory, check_max_similarity
import asyncio


async def process_and_store_memories():
    history = get_history()
    insights = get_insights(history)
    new_summarised_points_user = insights[1]
    new_summarised_points_ai = insights[0]
    
    # ✅ Deduplicate USER insights (facts about the user)
    for summarised_point in new_summarised_points_user:
        similar_facts, org_ids, scores = get_similar(summarised_point, 0.75)
        if similar_facts:
            max_similarity = check_max_similarity(scores)
            
            if max_similarity > 0.90:
                print(f"[Memory] Skipped User insight - already stored (similarity: {max_similarity:.2f})")
                continue
            
            decision = await deduplication_decision(summarised_point, similar_facts)
            result = merge_memory(decision, 'User', summarised_point, org_ids)
            print(f"[Memory] {result}")
        else:
            upsert_vector('memLayer', 'User', summarised_point)
            print(f"[Memory] Stored new User insight")
    
    # ✅ Deduplicate AI insights to avoid storing echoes
    for ai_insight in new_summarised_points_ai:
        similar_facts, org_ids, scores = get_similar(ai_insight, 0.75)
        if similar_facts:
            max_similarity = check_max_similarity(scores)
            
            if max_similarity > 0.90:
                print(f"[Memory] Skipped AI insight - already stored (similarity: {max_similarity:.2f})")
                continue
        
        upsert_vector('memLayer', 'AI', ai_insight)
        print(f"[Memory] Stored AI insight")


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
            asyncio.run(process_and_store_memories())

if __name__ == "__main__":
    chat()

