import os
import json
import base64
import time
from datetime import datetime
from typing import List, Optional
from PIL import Image, ImageDraw
import streamlit as st
from pydantic import BaseModel, Field
from groq import Groq
from tavily import TavilyClient

# ==============================================================================
# 1. Scientific Environmental Factors Data
# ==============================================================================
CO2_FACTORS = {
    "PET": {"co2": 18, "water": 1.2, "tree": 0.003},
    "Plastic": {"co2": 22, "water": 1.5, "tree": 0.004},
    "Can": {"co2": 32, "water": 2.1, "tree": 0.006},
    "Glass": {"co2": 25, "water": 0.8, "tree": 0.005},
    "Paper": {"co2": 10, "water": 0.5, "tree": 0.002},
    "Vinyl": {"co2": 8, "water": 0.3, "tree": 0.001},
    "Other": {"co2": 12, "water": 0.7, "tree": 0.002}
}

# ==============================================================================
# 2. Pydantic Model (Clean Schema)
# ==============================================================================
class SourceItem(BaseModel):
    title: str = Field(description="출처 기관명 또는 문서 제목")
    url: str = Field(description="원문 URL")

class XAIReasoning(BaseModel):
    visual_features: List[str] = Field(description="시각적 판단 근거 3가지")
    confidence_score: int = Field(description="AI 판단 신뢰도 (0~100)")

class RecyclingGuide(BaseModel):
    item_name: str = Field(description="정확한 품목명 (영문/한글 혼용 가능)")
    material: str = Field(description="재질 (PET, Plastic, Can, Glass, Paper, Vinyl, Other 중 선택)")
    category: str = Field(description="대분류")
    recycling_rate_pct: int = Field(description="재활용 가능률 (0~100)")
    xai_reasoning: XAIReasoning = Field(description="AI 판단 근거")
    steps: List[str] = Field(description="실행형 3단계 배출 절차")
    cautions: List[str] = Field(description="주의사항")
    sdg_impact: str = Field(description="SDG 기여도 요약")
    local_note: Optional[str] = Field(default=None, description="지역 수거 특이사항")
    sources: List[SourceItem] = Field(default=[], description="근거 정보 출처")


# ==============================================================================
# 3. Core AI Engine
# ==============================================================================
class RecyclingService:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    def search_local_rules(self, query: str) -> list[dict]:
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=2,
                include_domains=["go.kr", "or.kr", "seoul.go.kr", "keco.or.kr"]
            )
            return response.get("results", [])
        except Exception:
            return []

    def draw_apple_bbox(self, image: Image.Image, label: str) -> Image.Image:
        """Apple Style Minimalist Bounding Overlay"""
        img_copy = image.copy().convert("RGB")
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        
        left, top, right, bottom = w * 0.18, h * 0.18, w * 0.82, h * 0.82
        
        # Ultra-thin clean line
        draw.rectangle([left, top, right, bottom], outline="#34C759", width=3)
        return img_copy

    def encode_image_to_base64(self, image: Image.Image) -> str:
        import io
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze(self, item_text: Optional[str] = None, image: Optional[Image.Image] = None, location: str = "전국 공통") -> RecyclingGuide:
        system_instruction = f"""
        You are EcoLens Intelligence, an enterprise AI system for recycling analysis.
        Analyze input and respond STRICTLY in JSON according to this schema.
        {json.dumps(RecyclingGuide.model_json_schema(), ensure_ascii=False, indent=2)}
        """

        search_results = self.search_local_rules(f"{location} {item_text or '폐기물'} 분리배출 규정")
        context_str = "\n".join([f"- {r['title']}: {r['content']}" for r in search_results])

        user_content = [{"type": "text", "text": f"Location: {location}\nItem: {item_text or 'See Image'}\nRules:\n{context_str}"}]

        if image:
            base64_image = self.encode_image_to_base64(image)
            user_content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            model_name = "llama-3.2-11b-vision-preview"
        else:
            model_name = "llama-3.3-70b-versatile"

        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_content}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        return RecyclingGuide.model_validate_json(response.choices[0].message.content)


# ==============================================================================
# 4. Streamlit UI Architecture (Apple & Linear Aesthetic)
# ==============================================================================

st.set_page_config(
    page_title="EcoLens — AI Recycling Intelligence",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apple/Geist Typography & Ultra-clean Minimalist CSS
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/font-geist/latest/geist.css');
    
    /* 1. Global Minimal Canvas */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F5F5F7 !important;
        color: #1D1D1F !important;
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif !important;
        letter-spacing: -0.011em;
    }

    /* Remove Streamlit Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 720px !important;
    }

    /* 2. Apple Style Top Navbar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        margin-bottom: 60px;
        border-bottom: 1px solid rgba(0,0,0,0.06);
    }
    .nav-logo {
        font-weight: 600;
        font-size: 1.05rem;
        color: #1D1D1F;
        letter-spacing: -0.02em;
    }
    .nav-links {
        display: flex;
        gap: 24px;
        font-size: 0.85rem;
        color: #86868B;
    }
    .nav-btn {
        background: #1D1D1F;
        color: #FFFFFF !important;
        padding: 6px 14px;
        border-radius: 980px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* 3. Hero Section Typography */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 600;
        line-height: 1.08;
        letter-spacing: -0.03em;
        color: #1D1D1F;
        margin-bottom: 12px;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #86868B;
        font-weight: 400;
        margin-bottom: 40px;
        line-height: 1.4;
    }

    /* 4. Streamlit Widget Customization (Hide Ugly Streamlit Borders) */
    .stTextInput>div>div>input {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 14px !important;
        height: 52px !important;
        font-size: 0.95rem !important;
        padding-left: 16px !important;
        color: #1D1D1F !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
    }
    
    .stSelectbox>div>div {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 14px !important;
        height: 52px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
    }

    /* 5. Minimal Apple Button */
    div.stButton > button {
        background-color: #1D1D1F !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 14px !important;
        height: 52px !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        transition: opacity 0.2s ease !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        opacity: 0.88 !important;
    }

    /* 6. Spec Sheet Card (Linear / Apple SaaS Style) */
    .spec-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 32px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        margin-top: 40px;
        border: 1px solid rgba(0,0,0,0.04);
        opacity: 0;
        animation: fadeIn 0.3s ease-in-out forwards;
    }

    @keyframes fadeIn {
        to { opacity: 1; }
    }

    .accent-green {
        color: #34C759;
        font-weight: 600;
    }

    .spec-row {
        display: flex;
        justify-content: space-between;
        padding: 14px 0;
        border-bottom: 1px solid #F5F5F7;
        font-size: 0.95rem;
    }
    .spec-label { color: #86868B; }
    .spec-value { color: #1D1D1F; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# API Keys
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")

# Navbar
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">EcoLens</div>
    <div class="nav-links">
        <span>Intelligence</span>
        <span>SDGs</span>
        <span>Enterprise</span>
        <span class="nav-btn">Analyze</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("GROQ & TAVILY API Keys required.")
    st.stop()

service = RecyclingService(groq_api_key=GROQ_API_KEY, tavily_api_key=TAVILY_API_KEY)

# Hero Title (Apple Style)
st.markdown('<div class="hero-title">AI Recycling<br>Intelligence.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Instantly analyze items, material taxonomy, and regional compliance with precision vision AI.</div>', unsafe_allow_html=True)

# Input Section (Clean & Minimal)
tab_file, tab_name = st.tabs(["Upload Image", "Search Item"])
uploaded_image = None
item_text = None

with tab_file:
    img_file = st.file_uploader("Upload Item Image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if img_file:
        uploaded_image = Image.open(img_file)
        st.image(uploaded_image, use_container_width=True)

with tab_name:
    item_text = st.text_input("Item Name", placeholder="e.g. Plastic Bottle, Milk Carton", label_visibility="collapsed")

location = st.selectbox("Location", ["전국 공통", "서울특별시 강남구", "서울특별시 마포구", "경기도 수원시", "부산광역시 해운대구"], label_visibility="collapsed")

st.write("")
analyze_btn = st.button("Continue →")

# ------------------------------------------------------------------------------
# Analysis Result (SaaS Spec Sheet Layout)
# ------------------------------------------------------------------------------
if analyze_btn:
    if not uploaded_image and not item_text:
        st.warning("Please upload an image or enter an item name.")
    else:
        with st.spinner("Processing analysis..."):
            try:
                guide = service.analyze(item_text=item_text, image=uploaded_image, location=location)

                # 1. Bounding Box Image Display (if image uploaded)
                if uploaded_image:
                    st.write("")
                    bbox_img = service.draw_apple_bbox(uploaded_image, label=guide.item_name)
                    st.image(bbox_img, use_container_width=True)

                # 2. Linear/Apple Style Clean Spec Card
                st.markdown(f"""
                <div class="spec-card">
                    <div style="font-size: 0.8rem; color: #86868B; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px;">AI Analysis Complete</div>
                    <div style="font-size: 2rem; font-weight: 600; color: #1D1D1F; margin-bottom: 20px;">{guide.item_name}</div>
                    
                    <div class="spec-row">
                        <span class="spec-label">Recyclability Rate</span>
                        <span class="spec-value accent-green">{guide.recycling_rate_pct}%</span>
                    </div>
                    <div class="spec-row">
                        <span class="spec-label">Material</span>
                        <span class="spec-value">{guide.material}</span>
                    </div>
                    <div class="spec-row">
                        <span class="spec-label">Category</span>
                        <span class="spec-value">{guide.category}</span>
                    </div>
                    <div class="spec-row">
                        <span class="spec-label">AI Confidence Score</span>
                        <span class="spec-value">{guide.xai_reasoning.confidence_score}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 3. Steps Block
                st.write("")
                st.markdown("##### Disposal Protocol")
                for i, step in enumerate(guide.steps, 1):
                    st.markdown(f"**0{i}** &nbsp;&nbsp; {step}")

                # 4. Impact
                mat_key = guide.material if guide.material in CO2_FACTORS else "Other"
                impact = CO2_FACTORS[mat_key]

                st.write("")
                st.markdown(f"""
                <div style="background:#FFFFFF; padding:24px; border-radius:16px; border:1px solid rgba(0,0,0,0.04); margin-top:20px;">
                    <div style="font-size:0.85rem; color:#86868B; margin-bottom:8px;">ENVIRONMENTAL IMPACT</div>
                    <div style="display:flex; gap:32px;">
                        <div>
                            <div style="font-size:1.4rem; font-weight:600; color:#1D1D1F;">{impact['co2']}g</div>
                            <div style="font-size:0.8rem; color:#86868B;">CO₂ Saved</div>
                        </div>
                        <div>
                            <div style="font-size:1.4rem; font-weight:600; color:#1D1D1F;">{impact['water']}L</div>
                            <div style="font-size:0.8rem; color:#86868B;">Water Preserved</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 5. SDG Footnote
                st.write("")
                st.caption(f"🌍 **UN SDGs Compliance:** {guide.sdg_impact}")

            except Exception as e:
                st.error(f"Analysis error: {e}")

# Bottom Minimal Landing Footer
st.write("")
st.write("")
st.markdown("""
<div style="border-top: 1px solid rgba(0,0,0,0.06); padding-top: 32px; font-size: 0.8rem; color: #86868B; display: flex; justify-content: space-between;">
    <span>EcoLens AI © 2026</span>
    <span>Powered by Groq & UN SDG Framework</span>
</div>
""", unsafe_allow_html=True)
