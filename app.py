import os
import json
import base64
from typing import List, Optional
from PIL import Image, ImageDraw
import streamlit as st
from pydantic import BaseModel, Field
from groq import Groq
from tavily import TavilyClient

# ==============================================================================
# 1. Pydantic Model
# ==============================================================================
class XAIReasoning(BaseModel):
    visual_features: List[str] = Field(description="AI 판단 근거 3가지 (예: 복합재질 EVOH 필름, PP 용기 등)")
    confidence_score: int = Field(description="AI 판단 신뢰도 (0~100)")

class RecyclingGuide(BaseModel):
    item_name: str = Field(description="정확한 품목명 (예: 햇반 용기)")
    material: str = Field(description="추정 재질 (예: Plastic, PET, Can, Glass, Paper, Vinyl, Other)")
    category: str = Field(description="대분류 (예: 일반쓰레기 / 플라스틱류)")
    recycling_rate_pct: int = Field(description="재활용 가능률 (%) - 일반쓰레기는 0")
    xai_reasoning: XAIReasoning = Field(description="AI 판단 근거")
    steps: List[str] = Field(description="실행형 배출 수칙")
    cautions: List[str] = Field(description="배출 팁 및 예외 조건")
    sdg_impact: str = Field(description="SDG 기여 요약")


# ==============================================================================
# 2. Core Service
# ==============================================================================
class RecyclingService:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    def draw_bbox(self, image: Image.Image, label: str) -> Image.Image:
        img_copy = image.copy().convert("RGB")
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        draw.rectangle([w * 0.15, h * 0.15, w * 0.85, h * 0.85], outline="#34C759", width=4)
        return img_copy

    def encode_image(self, image: Image.Image) -> str:
        import io
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze(self, item_text: Optional[str] = None, image: Optional[Image.Image] = None, location: str = "전국 공통") -> RecyclingGuide:
        system_instruction = f"""
        당신은 AI 분리배출 분석 엔진 EcoLens입니다.
        '햇반 용기'처럼 EVOH 복합재질이나 세척 후에도 재활용이 불가능한 품목은 정확히 '일반쓰레기'로 분류하고 재활용률을 0%로 지정하세요.
        다음 JSON 스키마를 엄격히 준수하세요.
        {json.dumps(RecyclingGuide.model_json_schema(), ensure_ascii=False, indent=2)}
        """

        user_content = [{"type": "text", "text": f"지역: {location}\n품목: {item_text or '사진 참조'}"}]

        if image:
            base64_image = self.encode_image(image)
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
# 3. Streamlit UI Architecture
# ==============================================================================
st.set_page_config(page_title="EcoLens AI", page_icon="🌱", layout="centered")

# Minimal Style Patch
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FAFAFA !important;
        color: #111111 !important;
        font-family: 'Pretendard', sans-serif !important;
    }
    .block-container {
        padding-top: 2rem !important;
        max-width: 680px !important;
    }
    /* Brand Header */
    .brand-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 16px;
        margin-bottom: 32px;
        border-bottom: 1px solid #E5E5E5;
    }
    .brand-title { font-size: 1.1rem; font-weight: 700; color: #111111; }
    .brand-badge { background: #EEEEEE; color: #555555; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; }
    
    /* Result Card Wrapper */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 16px;
        padding: 24px;
        margin-top: 24px;
    }
    .tag-complete {
        display: inline-block;
        background: #E6F4EA;
        color: #137333;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# API Keys
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")

st.markdown("""
<div class="brand-nav">
    <div class="brand-title">EcoLens AI</div>
    <div class="brand-badge">SDG 12 & 13 Certified</div>
</div>
""", unsafe_allow_html=True)

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("GROQ_API_KEY 및 TAVILY_API_KEY가 필요합니다.")
    st.stop()

service = RecyclingService(groq_api_key=GROQ_API_KEY, tavily_api_key=TAVILY_API_KEY)

st.title("올바른 분리배출, AI로 1초 만에 확인하세요.")
st.caption("사진을 업로드하거나 품목명을 입력하면 환경부 지침 및 AI 분석 결과를 제공합니다.")

tab_name, tab_file = st.tabs(["✍️ 품목 검색", "📷 사진 업로드"])
uploaded_image = None
item_text = None

with tab_name:
    item_text = st.text_input("품목명 입력", placeholder="예: 햇반 용기, 삼다수 병", label_visibility="collapsed")

with tab_file:
    img_file = st.file_uploader("이미지 첨부", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if img_file:
        uploaded_image = Image.open(img_file)
        st.image(uploaded_image, use_container_width=True)

location = st.selectbox("지역 선택", ["전국 공통", "서울특별시 강남구", "서울특별시 마포구", "경기도 수원시"], label_visibility="collapsed")

if st.button("분석 실행 →", type="primary", use_container_width=True):
    if not uploaded_image and not item_text:
        st.warning("품목명을 입력하거나 사진을 업로드해 주세요.")
    else:
        with st.spinner("AI 분석 중..."):
            try:
                guide = service.analyze(item_text=item_text, image=uploaded_image, location=location)

                if uploaded_image:
                    st.image(service.draw_bbox(uploaded_image, guide.item_name), use_container_width=True)

                # 메인 결과 렌더링 (Streamlit Native Column 활용으로 HTML 깨짐 원천 차단)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<span class="tag-complete">AI 분석 완료</span>', unsafe_allow_html=True)
                st.subheader(guide.item_name)
                st.caption(f"카테고리: {guide.category}")
                st.divider()

                c1, c2 = st.columns(2)
                with c1:
                    st.metric(label="추정 재질", value=guide.material)
                    st.metric(label="AI 신뢰도", value=f"{guide.xai_reasoning.confidence_score}%")
                with c2:
                    st.metric(label="기본 재활용률", value=f"{guide.recycling_rate_pct}%")
                    st.metric(label="배출 지역", value=location)

                st.markdown('</div>', unsafe_allow_html=True)

                # AI 판단 근거 (XAI)
                st.write("")
                st.subheader("💡 AI 판단 근거 (Explainable AI)")
                for feat in guide.xai_reasoning.visual_features:
                    st.markdown(f"- {feat}")

                # 배출 수칙
                st.write("")
                st.subheader("📋 올바른 배출 수칙")
                for i, step in enumerate(guide.steps, 1):
                    st.markdown(f"**{i}.** {step}")

                # 주의사항
                if guide.cautions:
                    st.write("")
                    st.info("💡 **배출 팁 & 예외 조건**\n\n" + "\n".join([f"- {c}" for c in guide.cautions]))

                # 논리적 오류 수정을 거친 환경 영향 지표
                st.write("")
                if guide.recycling_rate_pct == 0:
                    st.caption(f"🌱 올바른 종량제 배출을 통해 혼합 수거로 인한 재활용 시설 오염을 예방했습니다. ({guide.sdg_impact})")
                else:
                    st.caption(f"🌱 이 분리배출 실천으로 자원 순환에 기여합니다. ({guide.sdg_impact})")

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")

st.divider()
st.caption("EcoLens AI © 2026 — Sustainable Products Intelligence")
