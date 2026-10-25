# 🖼️ AI-Powered Image Organizer

<p align="center">
  <img src="docs/images/project-cover.jpg" alt="Local Vision Agent — autonomous ai" width="720" />
</p>

> Automatically classify and organize thousands of photos using local LLM vision models and intelligent agents

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-LLaMA3.2-orange.svg)](https://ollama.ai/)

## 🎯 Description

Managing large photo collections is time-consuming and tedious. Whether you're a:
- **Content Creator** drowning in thousands of unsorted media files
- **Digital Marketer** needing to categorize campaign assets
- **Photographer** organizing client shoots
- **Data Scientist** preparing image datasets
- **Small Business Owner** managing product photos

This tool automatically analyzes and organizes your images by content using AI vision models—all running **locally** with complete privacy.

## ✨ Key Features

- 🤖 **AI Vision Classification** - Uses LLaMA 3.2 Vision to understand image content
- 🔄 **Autonomous Agent** - Self-organizes entire directories without manual intervention
- 🏷️ **Smart Categorization** - Recognizes travel, food, family, pets, work, nature, sports, events, and more
- 🔒 **100% Local & Private** - No cloud APIs, your photos never leave your machine
- 📁 **Batch Processing** - Handles hundreds of images automatically
- 🛠️ **Production Ready** - Built with LangChain for reliability and extensibility

## 🚀 Real-World Applications

### For Businesses
- **E-commerce**: Auto-organize product images by category for faster catalog management
- **Marketing Agencies**: Sort campaign assets and client deliverables efficiently
- **Real Estate**: Categorize property photos (interior, exterior, amenities)
- **Content Libraries**: Maintain organized media databases for teams

### For Individuals
- **Travel Bloggers**: Automatically separate travel photos by type
- **Event Photographers**: Sort wedding/event photos by category
- **Digital Decluttering**: Clean up years of unorganized photo collections
- **Dataset Preparation**: Organize images for ML training or computer vision projects

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     User Input                           │
│              (Source & Destination Dirs)                 │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              LangChain Agent Executor                    │
│              (llama3-groq-tool-use)                      │
│                                                          │
│  Orchestrates workflow using available tools:            │
│   1. list_images() → get all images                      │
│   2. classify_image() → categorize content               │
│   3. create_directory() → make category folders          │
│   4. move_file() → relocate to category                  │
│   5. Loop until list_images() returns []                 │
└────┬────────────────────────────────────┬────────────────┘
     │                                    │
     │ Calls classify_image()             │ Calls file tools
     ▼                                    ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   LLaMA 3.2 Vision       │  │  FileManagementToolkit   │
│   (Image Classifier)     │  │                          │
│                          │  │  • move_file()           │
│  • Base64 decode image   │  │  • copy_file()           │
│  • Analyze content       │  │  • list_directory()      │
│  • Return category       │  │  • read_file()           │
└──────────────────────────┘  └──────────────────────────┘
            │                              │
            └──────────┬───────────────────┘
                       ▼
            ┌─────────────────────┐
            │  Organized Output   │
            │  organized/         │
            │    ├── travel/      │
            │    ├── food/        │
            │    ├── family/      │
            │    └── ...          │
            └─────────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required models pulled:
  ```bash
  ollama pull llama3.2-vision
  ollama pull llama3-groq-tool-use
  ```

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/marreddysainikhilreddy/personal-assistant-agent.git
   cd ai-image-organizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your images**
   ```bash
   mkdir photos organized
   # Copy your unsorted images to ./photos
   ```

## 💻 Usage

```bash
python image_organizer.py
```

Follow the interactive prompts:
1. Enter source folder (e.g., `./photos`)
2. Enter destination folder (e.g., `./organized`)
3. Type `ORGANIZE` to start automatic organization

The agent will:
- 🔍 Scan all images in source directory
- 🧠 Classify each image using AI vision
- 📂 Create category folders automatically
- ✅ Move images to appropriate categories
- 🔄 Continue until all images are organized

## 🛠️ Technical Highlights

### Technologies Used
- **LangChain**: Agent orchestration and tool management
- **Ollama**: Local LLM inference
- **LLaMA 3.2 Vision**: State-of-the-art vision model for image understanding
- **PIL (Pillow)**: Image processing and encoding

### Design Patterns
- **Tool-based Agent Architecture**: Modular, extensible design
- **Autonomous Workflow**: Self-correcting agent that verifies completion
- **Base64 Image Encoding**: Efficient image transmission to vision model

## 📊 Supported Image Formats

- JPEG/JPG
- PNG
- WEBP

## 🎯 Categories Detected

The system recognizes 10+ categories out of the box:
- 🌍 Travel & Landmarks
- 🍕 Food & Dining
- 👨‍👩‍👧‍👦 Family & People
- 🐕 Pets & Animals
- 💼 Work & Professional
- 🌲 Nature & Landscapes
- ⚽ Sports & Activities
- 🎉 Events & Celebrations
- 🤳 Selfies & Portraits
- 📄 Documents & Screenshots

*Easy to extend with custom categories!*

## 🔮 Future Enhancements

- [ ] Web UI for drag-and-drop organization
- [ ] Support for video files
- [ ] Custom category training
- [ ] Duplicate image detection
- [ ] Cloud storage integration (S3, Google Drive)
- [ ] Batch processing progress bar
- [ ] Face recognition for person-specific folders

## 🤝 Contributing

Contributions are welcome! This project is perfect for:
- Adding new classification categories
- Improving agent prompts
- Building a web interface
- Optimizing performance for large datasets

## 📝 License

MIT License - feel free to use in personal or commercial projects

## 💡 How this project is helpful

In an era where we generate millions of photos daily, manual organization is no longer scalable. This project demonstrates:
- **Practical AI Applications**: Real-world use of vision models beyond demos
- **Agent-Based Automation**: Self-directed systems that complete complex workflows
- **Privacy-First Design**: Local processing without compromising on AI capabilities
- **Production-Ready Code**: Professional patterns using industry-standard frameworks

---

**Built with ❤️ using LangChain, LLaMA, and Ollama**

⭐ Star this repo if you find it useful!
