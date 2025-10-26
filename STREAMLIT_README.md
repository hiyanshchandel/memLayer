# 🧠 MemLayer - Streamlit UI

A beautiful, modern chat interface for MemLayer with persistent memory capabilities.

## 🎨 Features

- **Beautiful Gradient UI**: Modern purple gradient design with smooth animations
- **Real-time Chat**: Instant messaging with the AI assistant
- **Memory Tracking**: Visual indicators showing when memories are being synced
- **Statistics Dashboard**: Track messages exchanged and memories stored
- **Auto-sync Progress**: Visual progress bar showing when the next memory sync will occur
- **Responsive Design**: Works beautifully on different screen sizes

## 🚀 Getting Started

### Prerequisites

Make sure you have the virtual environment activated and all dependencies installed:

```bash
source memlayer/bin/activate
pip install -r requirements.txt
```

### Running the App

To start the Streamlit interface:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

### Alternative: Command Line Interface

If you prefer the traditional command-line chat:

```bash
python chat.py
```

## 📊 How It Works

1. **Chat Interface**: Type your messages in the chat input at the bottom
2. **Auto-sync**: Every 4 messages, the system automatically processes and stores memories
3. **Memory Types**: 
   - 🧑 User insights (facts about you)
   - 🤖 AI insights (conversation context)
4. **Deduplication**: Similar memories are automatically merged to avoid redundancy

## 🎯 UI Components

### Main Chat Area
- Clean, modern chat interface with user and AI avatars
- Smooth message animations
- Welcome screen for new conversations

### Sidebar Stats
- **Messages Exchanged**: Total count of user + AI messages
- **Memories Stored**: Number of insights saved to memory
- **Last Sync**: Timestamp of the last memory synchronization
- **Auto-Sync Progress**: Visual indicator of when next sync occurs

### Controls
- **Clear Chat**: Reset the conversation (memories are preserved)
- **About Section**: Learn more about MemLayer capabilities

## 🛠️ Configuration

The app uses the same configuration as the CLI version:
- `.env` file for API keys and settings
- `config.py` for model configurations
- All existing chat functionality from `chat.py` is preserved

## 💡 Tips

- The app remembers context across your conversation
- Every 4 messages, memories are automatically processed
- Check the sidebar to see your conversation statistics
- Use the clear chat button to start fresh while keeping your memories

## 🔧 Troubleshooting

### Port Already in Use
If port 8501 is already in use, run:
```bash
streamlit run app.py --server.port 8502
```

### Module Not Found
Ensure you're in the virtual environment:
```bash
source memlayer/bin/activate
```

## 🎨 Customization

You can customize the UI by modifying the CSS in `app.py`:
- Colors: Change the gradient colors in the `linear-gradient` sections
- Layout: Adjust the `layout` parameter in `st.set_page_config()`
- Theme: Modify the custom CSS styles

## 📝 File Structure

- `app.py` - Streamlit UI implementation (NEW)
- `chat.py` - Original CLI chat interface (unchanged)
- `get_response.py` - Chat response logic
- `upsert_to_qdrant.py` - Memory storage
- `deduplication.py` - Memory deduplication

---

Made with ❤️ using Streamlit
