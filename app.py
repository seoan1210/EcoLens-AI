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
# 1. 자원 절감 정밀 데이터
# ==============================================================================
CO2_FACTORS = {
    "PET": {"co2": 18, "water": 1.2},
    "Plastic": {"co2": 22, "water": 1.5},
    "Can": {"co2": 32, "water": 2.1},
    "Glass": {"co2": 25, "water": 0.8},
    "Paper": {"co2": 10, "water": 0.5},
    "Vinyl": {"co2": 8, "water": 0.3},
    "Other": {"co2": 12, "water": 0.7}
}

# ==============================================================================
# 2. Pydantic Model (XAI & 예외처리 강화)
# ==============================================================================
class XAIReasoning(BaseModel):
    visual_features: List[str] = Field(description="AI 판단 근거 3가지 (예: 복합재질 EVOH 필름, 밥알 잔여물, PP 용기 등)")
    confidence_score: int = Field(description="AI 판단 신뢰도 (0~100)")

class RecyclingGuide(BaseModel):
    item_name: str = Field(description="정확한 품목명 (예: 햇반 용기)")
    material: str = Field(description="재질 (PET, Plastic, Can, Glass, Paper, Vinyl, Other 중 선택)")
    category: str = Field(description="대분류 (예: 플라스틱류 / 일반쓰레기)")
    recycling_rate_pct: int = Field(description="기본 재활용 가능률 (%)")
    xai_reasoning: XAIReasoning = Field(description="AI 판단 근거")
    steps: List[str] = Field(description="실행형 배출 절차")
    cautions: List[str] = Field(description="주의사항 및 깨끗이 세척 시 재활용 가능 여부")
    sdg_impact: str = Field(description="SDG 기여도 요약")

# ==============================================================================
# 3. Core AI Engine
# ==============================================================================
class RecyclingService:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    def draw_apple_bbox(self, image: Image.Image, label: str) -> Image.Image:
        img_copy = image.copy().convert("RGB")
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        left, top, right, bottom = w * 0.15, h * 0.15, w * 0.85, h * 0.85
        draw.rectangle([left, top, right, bottom], outline="#34C759", width=4)
        return img_copy

    def encode_image_to_base64(self, image: Image.Image) -> str:
        import io
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze(self, item_text: Optional[str] = None, image: Optional[Image.Image] = None, location: str = "전국 공통") -> RecyclingGuide:
        system_instruction = f"""
        당신은 대한민국 대표 AI 분리배출 분석 엔진 EcoLens입니다.
        사용자가 '햇반' 같은 복합재질이나 모호한 제품을 입력해도, 단순 일반쓰레기 처리가 아니라 
        "왜 재활용이 어려운지(EVOH 등 복합재질)", "세척 후 배출 시 플라스틱 재활용 가능 여부" 등을 명확히 설명하세요.
        응답은 다음 JSON 스키마를 엄격히 준수하세요.
        {json.dumps(RecyclingGuide.model_json_schema(), ensure_ascii=False, indent=2)}
        """

        user_content = [{"type": "text", "text": f"지역: {location}\n품목: {item_text or '사진 참조'}"}]

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
# 4. Streamlit UI Architecture (Linear / Apple Dark-Light Premium)
# ==============================================================================

st.set_page_config(
    page_title="EcoLens Intelligence",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Clean Minimalist Styling
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FAFAFA !important;
        color: #111111 !important;
        font-family: 'Pretendard', -apple-system, sans-serif !important;
    }

    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 5rem !important;
        max-width: 680px !important;
    }

    /* Top Brand Nav */
    .brand-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 24px;
        margin-bottom: 40px;
        border-bottom: 1px solid #E5E5E5;
    }
    .brand-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #111111;
    }
    .brand-badge {
        background: #EEEEEE;
        color: #555555;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
    }

    /* Hero Headline */
    .hero-head {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.03em;
        color: #111111;
        margin-bottom: 10px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 36px;
        line-height: 1.5;
    }

    /* Input Box Customization */
    .stTextInput>div>div>input {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 12px !important;
        height: 50px !important;
        font-size: 0.95rem !important;
        color: #111111 !important;
    }
    
    div.stButton > button {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        height: 50px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }

    /* Dashboard Result Card */
    .result-container {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 20px;
        padding: 28px;
        margin-top: 32px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    .tag-green {
        display: inline-block;
        background: #E6F4EA;
        color: #137333;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin: 20px 0;
        padding: 16px 0;
        border-top: 1px solid #F0F0F0;
        border-bottom: 1px solid #F0F0F0;
    }
    .grid-item-label { font-size: 0.8rem; color: #777777; margin-bottom: 4px; }
    .grid-item-val { font-size: 1rem; font-weight: 600; color: #111111; }
</style>
""", unsafe_allow_html=True)

# API Keys
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")

# Brand Header
st.markdown("""
<div class="brand-nav">
    <div class="brand-title">EcoLens AI</div>
    <div class="brand-badge">SDG 12 & 13 Certified</div>
</div>
""", unsafe_allow_html=True)

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("GROQ_API_KEY 및 TAVILY_API_KEY 설정이 필요합니다.")
    st.stop()

service = RecyclingService(groq_api_key=GROQ_API_KEY, tavily_api_key=TAVILY_API_KEY)

# Hero Section
st.markdown('<div class="hero-head">올바른 분리배출,<br>AI로 1초 만에 확인하세요.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">사진을 업로드하거나 품목명을 입력하면 환경부 지침 및 AI 비전 분석 결과를 제공합니다.</div>', unsafe_allow_html=True)

tab_name, tab_file = st.tabs(["✍️ 품목 검색", "📷 사진 업로드"])
uploaded_image = None
item_text = None

with tab_name:
    item_text = st.text_input("품목명 입력", placeholder="예: 햇반 용기, 삼다수 병, 폐건전지", label_visibility="collapsed")

with tab_file:
    img_file = st.file_uploader("이미지 첨부", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if img_file:
        uploaded_image = Image.open(img_file)
        st.image(uploaded_image, use_container_width=True)

location = st.selectbox("지역 선택", ["전국 공통", "서울특별시 강남구", "서울특별시 마포구", "경기도 수원시"], label_visibility="collapsed")

st.write("")
analyze_btn = st.button("분석 실행 →")

# ------------------------------------------------------------------------------
# Result Render
# ------------------------------------------------------------------------------
if analyze_btn:
    if not uploaded_image and not item_text:
        st.warning("품목명을 입력하거나 사진을 업로드해 주세요.")
    else:
        with st.spinner("AI가 재질 및 분리배출 규정을 분석 중입니다..."):
            try:
                guide = service.analyze(item_text=item_text, image=uploaded_image, location=location)

                if uploaded_image:
                    st.write("")
                    bbox_img = service.draw_apple_bbox(uploaded_image, label=guide.item_name)
                    st.image(bbox_img, use_container_width=True)

                # 메인 결과 카드 (HTML 태그 노출 오류 전면 수정)
                st.markdown(f"""
                <div class="result-container">
                    <span class="tag-green">AI 분석 완료</span>
                    <h2 style="margin:0 0 8px 0; font-size:1.6rem; font-weight:700;">{guide.item_name}</h2>
                    <p style="color:#666666; font-size:0.9rem; margin:0;">카테고리: {guide.category}</p>
                    
                    <div class="grid-2">
                        <div>
                            <div class="grid-item-label">추정 재질</div>
                            <div class="grid-item-val">{guide.material}</div>
                        </div>
                        <div>
                            <div class="grid-item-label">기본 재활용률</div>
                            <div class="grid-item-val">{guide.recycling_rate_pct}%</div>
                        </div>
                        <div>
                            <div class="grid-item-label">AI 신뢰도</div>
                            <div class="grid-item-val">{guide.xai_reasoning.confidence_score}%</div>
                        </div>
                        <div>
                            <div class="grid-item-label">배출 지역</div>
                            <div class="grid-item-val">{location}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # AI XAI 상세 근거 (왜 햇반이 재활용이 어려운지 등의 심층 이유)
                st.write("")
                st.markdown("##### 💡 AI 판단 근거 (Explainable AI)")
                for feature in guide.xai_reasoning.visual_features:
                    st.markdown(f"• {feature}")

                # 배출 방법
                st.write("")
                st.markdown("##### 📋 올바른 배출 수칙")
                for i, step in enumerate(guide.steps, 1):
                    st.markdown(f"**{i}.** {step}")

                # 주의사항 / 예외 조건
                if guide.cautions:
                    st.write("")
                    st.info("💡 **배출 팁 & 예외 조건:**\n" + "\n".join([f"- {c}" for c in guide.cautions]))

                # 환경 영향
                mat_key = guide.material if guide.material in CO2_FACTORS else "Other"
                impact = CO2_FACTORS[mat_key]
                st.write("")
                st.caption(f"🌱 이 배출 실천으로 **CO₂ 약 {impact['co2']}g** 절감 및 **물 {impact['water']}L**가 보존됩니다. ({guide.sdg_impact})")

            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")

st.write("")
st.write("")
st.markdown("""
<div style="border-top: 1px solid #EEEEEE; padding-top: 20px; font-size: 0.8rem; color: #999999; text-align: center;">
    EcoLens AI © 2026 — Sustainable Products Intelligence
</div>
""", unsafe_allow_html=True)
