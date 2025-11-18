# chat_with_dj.py
import pandas as pd
import requests
import json

# --- Ollama Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b" # Or whatever model you prefer

# --- Data File ---
ANALYZED_CSV = 'my_library_analyzed.csv'

def ask_ollama(prompt):
    """Sends a prompt to the local Ollama instance and returns the response."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.8} # A little creativity
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to the DJ. Is Ollama running? ({e})"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def generate_library_summary(df):
    """Creates a concise summary of the library for the LLM's context."""
    total_songs = len(df)
    top_genres = df['genre'].value_counts().head(5)
    top_artists = df['artist'].value_counts().head(5)
    avg_year = df['year'].mean()
    
    summary = f"""
Here is a summary of the user's music library, which you are an expert on:
- Total Songs: {total_songs}
- Average Year of Music: {avg_year:.0f}
- Top 5 Genres: {', '.join([f"{genre} ({count} songs)" for genre, count in top_genres.items()])}
- Top 5 Artists: {', '.join([f"{artist} ({count} songs)" for artist, count in top_artists.items()])}
"""
    return summary

# --- Main Chat Loop ---
def start_chat():
    """Starts the interactive chat session."""
    try:
        df = pd.read_csv(ANALYZED_CSV)
    except FileNotFoundError:
        print(f"Error: The file '{ANALYZED_CSV}' was not found.")
        print("Please run 'analyze_library.py' first.")
        return

    library_summary = generate_library_summary(df)
    
    print("🎙️  The Wise DJ is online...")
    print("I've analyzed your library. Ask me anything about it.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        try:
            # Get user input
            user_question = input("You: ")

            # Check for exit condition
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("DJ: Keep the music playing. Peace out.")
                break
            
            # Construct the full prompt for Ollama
            # This is the most important part!
            full_prompt = f"""
You are a wise, witty, and deeply knowledgeable music DJ. You have just finished analyzing a person's entire music library and know it inside and out.

{library_summary}

Based on this summary and your broad musical knowledge, answer the following user question in a conversational and insightful way.

User Question: "{user_question}"

DJ Response:
"""
            
            print("DJ is thinking...")
            dj_response = ask_ollama(full_prompt)
            
            print(f"\nDJ: {dj_response}\n")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nDJ: Keep the music playing. Peace out.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")


if __name__ == "__main__":
    start_chat()