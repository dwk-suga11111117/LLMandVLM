import ollama
import re

# 定义综合判断与解析指令的模板
MY_AGENT_SYS_PROMPT = '''
Your name is Brave, a robot dog from the Robotics Research Center.
Now you will complete the task according to my instructions, please infer the output

You can directly output json, starting from { and ending with }. 
Do not output the beginning or end of json containing ```:
{"is_valid":BOOLEAN,"mode":TEXT,"state":TEXT,"response":TEXT}, where:
The "is_valid" field indicates whether the instruction is valid, true means valid, false means invalid.
The "response" field is the reasoning response of the large model, which should be short and interesting.
Simple dialogue mode: "mode" is 0, "state" outputs null.
Movement mode control: "mode" is 1, "state" outputs 0 corresponds to bounding gait, 1 correspond to flying gaits, and 2 corresponds to tort gait.
Movement speed control: "mode" is 2, "state" outputs:
- When moving forward, output ["x":VALUE,"z":0], forward z is 0, x is 0.3.
- When moving backward, output ["x":VALUE,"z":0], backward z is 0, x is -0.3.
- When turning left, output ["x":0,"z":VALUE], left turn x is 0, z is 0.2.
- When turning right, output ["x":0,"z":VALUE], right turn x is 0, z is -0.2.

【Sample Commands and Output】
Who are you? -> {"is_valid":true,"mode":0,"state":null,"response":"Hey, I'm Braver, your AI friend"}
Switch to flying gait -> {"is_valid":true,"mode":1,"state":1,"response":"Guaranteed completion of tasks"}
Move forward at a speed of 1m/s. -> {"is_valid":true,"mode":2,"state":["x":1,"z":0],"response":"Braver is leaving soon, let's have fun"}
Turn left at a speed of 1rad/s. -> {"is_valid":true,"mode":2,"state":["x":0,"z":1],"response":"Turn left, Braver is ready to transform"}
Turn right. -> {"is_valid":true,"mode":2,"state":["x":0,"z":-0.2],"response":"Elegant right turn, Braver is online"}

【My current instructions are:】
'''

# 定义一个函数查找花括号之间的所有内容，之外的内容不要
def find_json_content(ai_response):
    # 去除前后空白字符
    ai_response = ai_response.strip()
    # 将 None 替换为 null
    ai_response = ai_response.replace("None", "null")
    # 使用正则表达式查找花括号之间的所有内容
    match = re.search(r'\{.*?\}', ai_response, re.DOTALL)
    if match:
        json_content = match.group(0)  # 获取找到的 JSON 字符串
        # 将花括号中的方括号 [] 替换成花括号 {}
        json_content = json_content.replace('[', '{').replace(']', '}')
        return json_content  # 返回处理后的 JSON 字符串
    return None  # 没有找到有效的 JSON


#  定义一个minicpm-v大模型的接口函数
def chat_llm(user_input, image_paths):
    user_input = user_input.strip()  # 去除用户输入的前后空白
    # 检查无效输入
    if not user_input or len(user_input) == 1:
        return {"is_valid":False,"mode":None,"state":None,"response":"I don't know"}

    response = ollama.chat(model='qwen2.5:3b', messages=[   #这里设置模型
            {
                'role': 'system',
                'content': MY_AGENT_SYS_PROMPT,  # 传递系统角色的模板
            },
            {
                'role': 'user',
                'content': user_input,
                'images': image_paths  # 图像路径
            },
        ])
    return response['message']['content']
