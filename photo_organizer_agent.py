import os
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_agent
from typing import List

from pathlib import Path
import langchain

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from PIL import Image
import base64
from io import BytesIO


current_directory = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(current_directory, "photos")

INPUT_DIR = "./photos"
OUTPUT_DIR = "./organized"


file_toolkit = FileManagementToolkit(
    root_dir='.',
    # selected_tools=["list_directory", "move_file"]
)
file_tools = file_toolkit.get_tools()
# print(file_tools)

# For classification
llm_vision = ChatOllama(model="llama3.2-vision", temperature=0.2)

# For agent
# llm_agent = ChatOllama(model="llama3.2", temperature=0.2)
llm_agent = ChatOllama(model="llama3-groq-tool-use", temperature=0.2)


def encode_images_to_base64(image_path: str) -> str:
    with Image.open(image_path) as pil_image:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


@tool
def list_images(dir_path: str) -> List[str]:
    """ List all image files in the given directory path """
    extensions = {"jpg", "png", "jpeg", "webp"}
    path = Path(dir_path)
    images = []
    for extension in extensions:
        images.extend(path.glob(f"*.{extension}"))
        images.extend(path.glob(f"*.{extension.upper()}"))
    
    return [str(p) for p in sorted(images)]


@tool
def classify_image(image_path: str) -> str:
    """ Classify the content of the image at the given path into a category. """
    if not os.path.exists(image_path):
        return "Error: Image Not Found"
    
    base64_image = encode_images_to_base64(image_path)

    messages = [
        SystemMessage("""
            You are a precise photo classification assistant.
            
            Analyze the image carefully and classify it into ONE specific category.
            
            Common categories:
            - travel (landmarks, tourist locations, outdoor adventures)
            - food (meals, cooking, restaurants)
            - family (family gatherings, relatives)
            - pet (dogs, cats, animals)
            - work (office, meetings, professional events)
            - nature (landscapes, plants, wildlife)
            - sports (athletic activities, games)
            - event (parties, celebrations, weddings)
            - selfie (portrait, personal photos)
            - document (screenshots, receipts, text)
            
            CRITICAL RULES:
            - Output ONLY the category name in lowercase
            - NO explanations, NO punctuation, NO extra words
            - Be SPECIFIC - don't default to 'travel' for everything
        """),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": base64_image}},
        ])
    ]
    ai_response = llm_vision.invoke(messages)
    image_category = ai_response.content.strip()
    return image_category


# This function creates a folder if a folder doesn't exist
@tool
def create_directory(dir_path: str) -> str:
    """ Create a directory at the given path if it does not exist. """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return f"Directory ready: {path}"


@tool
def get_filename(path: str) -> str:
    """Return just the filename from a full path."""
    return os.path.basename(path)
    
agent_system_prompt = """
You are an image organization agent. Organize ALL images from source to destination by category.

 CRITICAL RULES:
1. ONLY output tool calls - NO text, NO explanations, NO "..." 
2. After EVERY move_file, immediately call list_images to check for more images
3. Continue until list_images returns []
4. ONLY when list_images returns [], output: "All images are organized successfully"

MANDATORY WORKFLOW - FOLLOW EXACTLY:
1. list_images(source_dir)
2. If images found:
   - classify_image(first_image_path)
   - create_directory(dest_dir/category)
   - move_file(source, destination)
   - IMMEDIATELY call list_images(source_dir) again
   - REPEAT from step 2
3. If list_images returns []: output "All images are organized successfully"

PATH FORMATS:
- Source: photos/{filename}
- Destination: organized/{category}/{filename}

 NEVER output text like "...", NEVER stop until list_images returns []
 ALWAYS call list_images after EVERY move_file operation

START NOW: Call list_images on the source directory.
"""

tools = file_tools + [list_images, classify_image, create_directory]

agent = create_agent(
    model=llm_agent,
    tools=tools,
    system_prompt=agent_system_prompt,
    debug=True,
)

# result = agent.invoke({
#     "messages": [{"role": "user", "content": "Start organizing ALL images now. organize from ./photos to ./organized"}]
# }, 
# config={"recursion_limit": 150})


def organize_images() :
    while True:
        user_input = input("Enter source folder (for images) ").strip()
        if user_input.lower() == "exit":
            break
        output_dir = input("Enter destination folder (for organized images): ").strip()
        if output_dir.lower() == "exit":
            break
        
        while True:
            user_command = input("Type ORGANIZE or 'back' to change folders: ").strip()
            if user_command.lower() == "exit":
                return
            if user_command.lower() == "back":
                break
            
            if user_command.lower() != "organize":
                print("Invalid command. Please type 'ORGANIZE', 'back' or 'exit'. ")
                continue

            result = agent.invoke({
                    "messages": [{"role": "user", "content": "Start organizing ALL images now. organize from ./photos to ./organized"}]},
                    config={"recursion_limit": 150}
            )

            print(result)
            break


if __name__ == "__main__":
    organize_images()
