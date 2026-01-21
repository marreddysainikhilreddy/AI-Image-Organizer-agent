import os
import base64
from typing import List
from pathlib import Path
from PIL import Image
from io import BytesIO

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
# from langchain.agents import 
from typing import List


file_toolkit = FileManagementToolkit(
    root_dir='.',
)
file_tools = file_toolkit.get_tools()

# For classification
llm_vision = ChatOllama(model="llama3.2-vision", temperature=0.2)
# llm_vision = ChatOllama(model="gpt-oss:20b", temperature=0.2)
# llm_vision = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", 
#     temperature=0.2,
#     google_api_key=""
# )

# For agent
# llm_agent = ChatOllama(model="llama3.2", temperature=0.2)
# llm_agent = ChatOllama(model="llama3-groq-tool-use", temperature=0.2)
llm_agent = ChatOllama(model="gpt-oss:20b", temperature=0.2)
# llm_agent = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", 
#     temperature=0.2,
#     google_api_key=""
# )



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
You are an image organizer. Move every image from source path to destination path.
Goal: organize the images from source path to the destination path in their respective category folders

RULES:
    - NEVER output any text, explanation, or placeholder.
    - ONLY respond with tool calls.
    - When list_images returns [], respond EXACTLY: "All images are organized successfully"


INSTRUCTIONS
    - Use list_images tool to get the list of all images file names in a directory
    - Use classify_image to get the category of an image 
    - Use create_directory to create a folder/directory
    - Use get_filename to get the file name
    - use move_file to move the file from a source path to destination path
    - use any other tools provided to you to achieve the goal

"""


tools = file_tools + [list_images, classify_image, create_directory]

agent = create_agent(
    model=llm_agent,
    tools=tools,
    system_prompt=agent_system_prompt,
    debug=True,
)


def organize_images() :
    while True:
        user_input_dir = input("Enter source folder (for your images) ").strip()
        if user_input_dir.lower() == "exit":
            break
        output_dir = input("Enter destination folder (for organized images): ").strip()
        if output_dir.lower() == "exit":
            break
        
        while True:
            user_command = input("I'm an AI agent over here. What task would you like me to perform?  Type 'organize' to start, 'back' to change folders, or 'exit': ").strip()
            if "exit" in user_command.lower():
                return
            if "back" in user_command.lower():
                break
            
            # print(user_command.lower())

            if "organize" not in user_command.lower() and "organizing" not in user_command.lower():
                print("Invalid command. Please type 'ORGANIZE', 'back' or 'exit'. ")
                continue

            result = agent.invoke({
                    "messages": [{"role": "user", "content": f"Start organizing ALL images now. organize from {user_input_dir} to {output_dir}"}]},
                    config={"recursion_limit": 150}
            )

            print(result)
            break


if __name__ == "__main__":
    organize_images()