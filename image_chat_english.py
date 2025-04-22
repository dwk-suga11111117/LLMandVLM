from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import time
import torch
from PIL import Image
import os
import time
import torch
from PIL import Image, ImageFile
import fcntl
from bitsandbytes import optim

ImageFile.LOAD_TRUNCATED_IMAGES = True

MY_AGENT_SYS_PROMPT1 = '''
You are a bipedal robot, and now you need to pass through the terrain ahead. Please conduct a professional analysis of the terrain in the picture. Requirements:

1. * * Terrain classification * * (prioritize identifying one of the following three categories):
-Grassland: a natural or artificially planted area covered with herbaceous plants
-Permeable brick pavement: a hardened pavement with a porous structure
-Asphalt pavement
-Porous pavement
-Sand road surface
-Rubber flooring
-Marble tiles
-Special terrain

2. * * Analysis Requirements * *:
-Firstly, summarize the overall terrain distribution in one sentence (such as' There is only one type of terrain ahead, with permeable brick pavement 'or' There are two types of terrain ahead, with grassland on the left and rock area on the right ')
-Expand detailed analysis according to the following dimensions:
Surface texture features (such as grass leaf density/brick seam width/graininess)
• Structural hierarchy (such as vegetation height stratification/pavement patterns)
Surface condition (damp/dry/damaged/covered, etc.)
• Terrain proportion (the proportion of this terrain in all terrains in the picture, which needs to be described with a specific numerical range, such as "proportion of about 80%")

3. * * Multi terrain scenes * * (if there are two or more types):
-First, explain the terrain distribution (such forward terrain is A and back terrain is B)
-Annotate spatial distribution characteristics (such as stripes/patches)
-Compare the differences in physical properties between different regions
-Identify the connection characteristics of the transition zone

4. * * Output Specification * *:
-Directly analyze the content without mentioning descriptions such as' blurry ',' unclear ',' limited visibility ', etc
-If there are two types of terrain, use [Terrain Type 1]/[Terrain Type 2] to divide them into sections
-When uncertain elements are present, use the phrase 'suspected...' and explain the basis
-All analysis items end with complete sentences

**Example instruction**
Little dog, what terrain is ahead?
**Example output format**
There are two types of terrain ahead, namely [A road/terrain] and [B road/terrain]. [A road/terrain] is on the left side, and [B road/terrain] is on the right side
A Road Surface/Terrain

-Surface texture features:【text】
-Structural hierarchy:【text】
-Surface condition:【text】
-Terrain proportion:【text】

B Road Surface/Terrain

-Surface texture features:【text】
-Structural hierarchy:【text】
-Surface condition:【text】
-Terrain proportion:【text】
'''

MY_AGENT_SYS_PROMPT2 = '''
Please analyze the terrain ahead as a bipedal robot and choose a suitable gait. requirement:
1. * * Terrain classification * *: Firstly, determine the terrain as "grassland", "permeable brick pavement", and "asphalt pavement".
2. * * Gait selection rules * *:
-Permeable brick terrain: using * * Gait A * * (jumping to avoid obstacles), with a speed of less than 0.5 m/s
-Special terrain: using * * Gait B* * (efficient and stable movement), with a speed of less than 0.5 m/s
-Grassland terrain (such as snow): Use * * Gait C * * (low center of gravity to prevent subsidence), with a speed of less than 0.5 m/s
3. * * Output format * *:
-If the terrain is single:
Front terrain: [classification result], suggested speed [XX m/s], suggested
Gait: [gait type], precautions: [brief explanation]“
-If the terrain is divided into left and right:
Left terrain: [classification result], recommended speed [XX m/s], recommended gait: [gait type], precautions: [brief explanation]
Right terrain: [classification result], recommended speed [XX m/s], recommended gait: [gait type], precautions: [brief explanation]
**Recommended path * *:
Recommended choice: [Forward/Left/Right], Reason: [Based on gait suitability or terrain risk analysis]
4. Provide advice in the tone of "Engineer Xiao Wang" (serious but with a touch of humanity)
5. Only output analysis results and recommendations without further explanation.

Output example (single terrain):
Front terrain: permeable brick road surface, recommended gait: flying， Speed: 0.4 m/s, Attention: Pay attention to the gap between the brick joints to avoid getting stuck

Output example (dual terrain):
Left terrain: grassland, recommended gait: A， Speed: 0.3 m/s, Attention: Avoid sinking due to dense grass leaves
Right terrain: permeable brick road surface, recommended gait: B， Speed: 0.5 m/s, Attention: If the brick surface is wet and slippery, lower the center of gravity
**Path recommendation * *: Prioritize the right side, reason: B gait has higher passability, and the friction coefficient of permeable brick pavement is better than that of grassland
'''

MY_AGENT_SYS_PROMPT3 = '''
You are a bipedal robotic dog with a camera in front of you. Now that you see this photo, please analyze it
'''

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/dwk/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained("/home/dwk/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct")

def safe_save_image(image_data):
    """Atomic image writing (integrate with camera code)"""
    temp_path = "/home/dwk/123/QWEN/picture/.temp.png"
    target_path = "/home/dwk/123/QWEN/picture/li4.png"

    try:
        with open(temp_path, "wb") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(image_data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target_path)
    finally:
        try:
            os.remove(temp_path)
        except:
            pass

def load_image_with_retry(image_path, max_retries=5, delay=0.1):
    """Image loading with retry and file lock"""
    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                img = Image.open(f)
                img.load()
                return img.convert("RGB")
        except (IOError, ImageFile.TruncatedImageError) as e:
            print(f"Image load failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(delay * (attempt + 1))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN) if 'f' in locals() else None

    raise RuntimeError(f"Failed to load image: {image_path}")

def process_query(user_input):
    """Process user query and generate response"""
    try:
        image = Image.open("/home/dwk/123/QWEN/picture/li4.png").convert("RGB")
        image = image.resize((640, 480))

        # 新增图片保存逻辑
        output_dir = "/home/dwk/123/QWEN/picture"
        base_num = 2

        while True:
            output_path = f"{output_dir}{base_num}.png"
            if not os.path.exists(output_path):
                break
            base_num += 1
            if base_num > 100:  # 防止无限循环
                raise ValueError("over")

        image.save(output_path)
        # print(f"分析图片已保存至：{output_path}")
        if "what" in user_input.lower():
            system_prompt = MY_AGENT_SYS_PROMPT1
        elif "analyze" in user_input.lower():
            system_prompt = MY_AGENT_SYS_PROMPT2
        else:
            system_prompt = MY_AGENT_SYS_PROMPT3

        messages = [
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_input},
            ]
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    except FileNotFoundError:
        return "Error: 1.png image file not found"
    except Exception as e:
        return f"Processing error: {str(e)}"

def main():
    print("System ready. Include 'braver' in commands (type 'exit' to quit)")
    while True:
        user_input = input("\nCommand: ").strip()

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        if 'braver' not in user_input.lower():
            print("Command missing 'braver' keyword")
            continue

        if not os.path.exists("/home/dwk/123/QWEN//picture/li4.png"):
            print("Error: 1.png not found")
            continue

        print("\nAnalyzing...")
        start_time = time.time()
        response = process_query(user_input)
        print(f"\nResponse ({time.time() - start_time:.1f}s):")
        print(response)

if __name__ == "__main__":
    main()