import streamlit as st
import json
import os
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# --- 1. 环境与网络配置 ---
load_dotenv()
#os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
#os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# 获取配置
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-sxfkcwytfxichqtoncobwxtjjiufigkcqadygvjflvyqkhsd").strip()
if "HTTP_PROXY" in os.environ: del os.environ["HTTP_PROXY"]
if "HTTPS_PROXY" in os.environ: del os.environ["HTTPS_PROXY"]
# --- 2. 页面配置 ---
st.set_page_config(page_title="丙午马年智能春联生成器", page_icon="🐎", layout="centered")

# --- 检查 Key 是否存在 ---
if not API_KEY:
    st.error("❌ 未检测到 API Key！请在同级目录下创建 .env.example 文件并配置 DEEPSEEK_API_KEY。")
    st.stop()

# --- 初始化大模型客户端 ---
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    st.error(f"客户端初始化失败: {e}")

# --- 自定义CSS ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 30%, #FFFBF0 0%, #FFE4E1 100%);
    }
    .title { 
        color: #D22B2B; 
        font-family: 'KaiTi', 'STKaiti', serif; 
        text-align: center; 
        font-size: 3.5em; 
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    /* 按钮美化 */
    .stButton > button {
        background: linear-gradient(to right, #e52d27, #b31217);
        border: 2px solid #FFD700 !important;
        color: #FFD700 !important;
        font-size: 22px !important;
        font-weight: bold !important;
        border-radius: 50px !important;
        padding: 10px 20px !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(229, 45, 39, 0.7) !important;
    }

    /* 横批样式 */
    .couplet-header {
        background: linear-gradient(180deg, #D41420 0%, #B00B15 100%);
        color: #FFD700;
        padding: 15px 40px;
        border: 3px solid #F6D365;
        border-radius: 8px;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        font-family: 'KaiTi', 'STKaiti', serif;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        letter-spacing: 10px;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }

    /* 上下联样式 (高度锁定版) */
    .couplet-vertical {
        background: linear-gradient(90deg, #D41420 0%, #B00B15 100%);
        color: #FFD700;

        /* 竖排核心 */
        writing-mode: vertical-rl;
        text-orientation: upright;

        /* 【关键修改】锁定高度，与图片对齐 */
        height: 550px; 

        /* 使用 Flex 让文字在长条里居中 */
        display: flex;
        align-items: center;    /* 水平居中 (在竖排模式下) */
        justify-content: space-evenly; /* 垂直居中 (如果想铺满整条，改成 space-evenly) */

        /* 边框与字体 */
        border: 3px solid #F6D365;
        border-radius: 8px;
        font-size: 42px; /* 字体加大，更饱满 */
        font-weight: bold;
        font-family: 'KaiTi', 'STKaiti', serif;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        letter-spacing: 8px; /* 字间距拉大，占满空间 */

        /* 容器居中 */
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)


# --- 加载 Embedding 模型 ---
@st.cache_resource
def load_embedding_model():
    local_model_path = './paraphrase-multilingual-MiniLM-L12-v2'
    if not os.path.exists(local_model_path):
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return SentenceTransformer(local_model_path)

model = load_embedding_model()

# --- 加载本地知识库 (已适配下拉框逻辑) ---
def load_knowledge_base():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            # 简单校验数据完整性
            if data and "category" in data[0]:
                return data
            else:
                return [{"category": "通用", "text": "万事如意"}]
    except (FileNotFoundError, json.JSONDecodeError):
        # 兜底数据
        return [{"category": "通用", "text": "万事如意"}]


knowledge_base = load_knowledge_base()


# --- RAG 检索模块 ---
def rag_retrieve_context(user_query):
    corpus_sentences = [item["text"] for item in knowledge_base]
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    best_match_index = int(cosine_scores.argmax())
    return corpus_sentences[best_match_index], float(cosine_scores[best_match_index])


# --- 新增：使用 SiliconFlow 生成图片 (国内直连) ---
def generate_image_siliconflow(prompt):
    url = "https://api.siliconflow.cn/v1/images/generations"

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    # 使用 FLUX.1-schnell 模型
    payload = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": prompt,
        "image_size": "512x1024"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            # SiliconFlow 返回的是一个图片 URL
            image_url = response.json()['data'][0]['url']
            return image_url
        else:
            print(f"绘图API报错: {response.text}")
            return None
    except Exception as e:
        print(f"请求异常: {e}")
        return None

# --- 核心生成逻辑 (含格律控制) ---
def generate_couplet(name, job, style):
    context_text, score = rag_retrieve_context(job)

    style_prompt_map = {
        "赛博朋克": "风格要求：使用科技感词汇（量子、比特），语气硬核。",
        "互联网黑话": "风格要求：使用大厂黑话（赋能、闭环），幽默诙谐。",
        "幽默搞怪": "风格要求：风趣幽默，可以玩梗。",
        "传统典雅": "风格要求：辞藻华丽，古风韵味浓厚。"
    }
    style_prompt = style_prompt_map.get(style, "")

    # System Prompt 强化：加入格律专家设定
    system_prompt = f"""
    你是一位精通《联律通则》的AI春联大师。请为用户创作一副【马年七言春联】。

    【用户信息】
    - 名字：{name}
    - 职业：{job}
    - RAG关键词：{context_text}

    【硬性格律要求】
    1. **字数**：上下联各七个字。
    2. **平仄规则（仄起平收）**：
       - 上联最后一个字必须是**仄声**（汉语拼音三声或四声）。
       - 下联最后一个字必须是**平声**（汉语拼音一声或二声）。
    3. **对仗**：词性要相对（名词对名词，动词对动词）。
    4. 横批的内容尽量做到和上联下联有一定的关联性。
    
    【名字处理逻辑（重要）】
    请尝试将用户名字【{name}】融入联中（藏头、藏尾或嵌在中间均可）。
    **决策原则**：
    - 如果名字容易融入且意境优美（如“明、华、伟”），请**务必融入**。
    - 如果名字过于口语化、生僻或融入后会破坏对联的通顺度与格律（如“哈基米”、“只有帅”），请**果断放弃融入名字**，优先保证对联的整体文学质量。
    - 不要为了藏头而写出不通顺的句子！

    {style_prompt}

    【输出格式】
    请直接输出结果，不要思考过程，格式严格如下：
    上联内容||下联内容||横批内容
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请严格按照平仄格律创作，开始！"}
            ],
            temperature=0.85,  # 稍微调高一点创造力
            stream=False
        )
        result_text = response.choices[0].message.content.strip()
        parts = result_text.split("||")

        if len(parts) >= 3:
            return {"up": parts[0], "down": parts[1], "center": parts[2], "rag_info": context_text}
        else:
            return {"up": "灵马奔腾送福来", "down": "格式解析出意外", "center": "再试一次", "rag_info": context_text}

    except Exception as e:
        return {"up": "API 连接失败", "down": "请检查网络配置", "center": "出错啦", "rag_info": str(e)}


# --- 界面 UI ---
st.markdown('<h1 class="title">🐎 2026 丙午 灵马送福</h1>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: -10px; margin-bottom: 20px;">基于 RAG 技术与 DeepSeek 的个性化春联系统</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input("你的名字或昵称", placeholder="例如：小明")

    with col2:
        try:
            job_options = sorted(list(set([item.get("category", "通用") for item in knowledge_base])))
        except Exception:
            job_options = ["通用"]

        # 替换 text_input 为 selectbox
        user_job = st.selectbox("选择你的身份场景", job_options)

    style_option = st.selectbox("选择春联风格", ["传统典雅", "互联网黑话", "赛博朋克", "幽默搞怪"])

if st.button("✨ 立即生成专属运势 ✨", use_container_width=True):
    if not user_name:
        st.warning("请先输入名字哦！")
    else:
        # 1. 生成文字
        with st.spinner(f"正在连接 DeepSeek 思考对联..."):
            res = generate_couplet(user_name, user_job, style_option)

        # 2. 生成图片 (在后台进行，不先展示)
        with st.spinner(f"正在调用 FLUX.1 绘制年画..."):
            base_prompt = "Chinese New Year, year of the horse, masterpiece, 8k, best quality"
            if style_option == "赛博朋克":
                base_prompt += ", cyberpunk, neon lights, mechanical horse"
            elif style_option == "传统典雅":
                base_prompt += ", traditional chinese ink painting, red paper, calligraphy"
            elif style_option == "互联网黑话":
                base_prompt += ", pixel art, coding horse, matrix background"
            elif style_option == "幽默搞怪":
                base_prompt += ", funny cartoon 3d render, cute horse"

            img_url = generate_image_siliconflow(base_prompt)

        # 3. 最终布局展示 (门神布局)
        st.balloons()

        # 第一行：横批 (居中)
        st.markdown(f'<div class="couplet-header">{res["center"]}</div>', unsafe_allow_html=True)

        # 第二行：下联 - 图片 - 上联 (左中右布局)
        # 注意：传统习俗中，面对大门，右边贴上联，左边贴下联。
        # 这里的 col1 是屏幕左边（对应下联），col3 是屏幕右边（对应上联）
        col_left, col_mid, col_right = st.columns([1, 2, 1])

        with col_left:
            st.markdown(f'<div class="couplet-vertical">{res["down"]}</div>', unsafe_allow_html=True)
            st.caption("下联")

        with col_mid:
            if img_url:
                # 【关键修改】这里强制图片高度为 550px，与 CSS 中的 .couplet-vertical 保持一致
                # object-fit: cover 保证图片填满框且不变形
                st.markdown(
                    f'<img src="{img_url}" style="width:100%; height:550px; object-fit:cover; border-radius:10px; border:4px solid #D22B2B; box-shadow: 0 5px 15px rgba(0,0,0,0.2);">',
                    unsafe_allow_html=True
                )
            else:
                st.error("图片生成失败")

        with col_right:
            st.markdown(f'<div class="couplet-vertical">{res["up"]}</div>', unsafe_allow_html=True)
            st.caption("上联")

        # 底部揭秘
        st.success(f"💡 RAG 匹配梗：{res['rag_info']}")