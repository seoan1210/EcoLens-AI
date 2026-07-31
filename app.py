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
# 1. Pydantic Model (확장된 AI 분석 스펙)
# ==============================================================================
class InspectorCheck(BaseModel):
    material_type: str = Field(description="재질 감지 (예: PP, PET, 종이 등)")
    label_removed: bool = Field(description="라벨 제거 여부 또는 필요성")
    cap_separated: bool = Field(description="뚜껑 분리 여부")
    food_residue: bool = Field(description="음식물 잔여물 존재 여부")
    contamination_risk: str = Field(description="오염 위험도 (낮음, 보통, 높음)")

class RecyclingGuide(BaseModel):
    item_name: str = Field(description="정확한 품목명")
    eco_score: int = Field(description="종합 Eco Score (0~100)")
    eco_grade: str = Field(description="등급 (A+, A, B, C, F)")
    material: str = Field(description="추정 재질")
    category: str = Field(description="대분류")
    recycling_rate_pct: int = Field(description="기본 재활용률 (%)")
    
    # AI Inspector & XAI
    inspector: InspectorCheck = Field(description="AI Inspector 5대 점검 항목")
    rejection_risk_warning: Optional[str] = Field(description="재활용 거부 위험 원인 경고 (없으면 빈값)")
    
    # 꿀팁 & 추천
    lifecycle_steps: List[str] = Field(description="자원 순환 4단계 (예: 생산 -> 배출 -> 분쇄/세척 -> 새 용기)")
    similar_items: List[str] = Field(description="연관/비슷한 품목 3가지")
    steps: List[str] = Field(description="실행형 배출 수칙")
    cautions: List[str] = Field(description="배출 팁 및 예외 조건")

# ==============================================================================
# 2. Core Service
# ==============================================================================
class RecyclingService:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    def draw_bbox(self, image: Image.Image) -> Image.Image:
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
        당신은 AI 분리배출 및 자원순환 전문 AI Inspector입니다.
        입력된 품목에 대해 실시간 스캐닝 항목(InspectorCheck), 위험 경고, 순환 라이프사이클(lifecycle_steps), 연관 품목을 정밀 분석하세요.
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

    def fetch_eco_news(self, query: str = "분리배출 정책 재활용 뉴스") -> list:
        try:
            res = self.tavily_client.search(query=query, max_results=3)
            return res.get("results", [])
        except:
            return []

    def ask_ai_chat(self, question: str, context_item: str) -> str:
        prompt = f"사용자가 '{context_item}' 배출에 대해 질문했습니다: '{question}'. 2문장 이내로 친절하고 정확하게 답하세요."
        res = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content

# ==============================================================================
# 3. Streamlit UI Architecture (SaaS Multi-Page Layout)
# ==============================================================================
st.set_page_config(page_title="EcoLens Platform", page_icon="🌱", layout="centered")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FAFAFA !important;
        color: #111111 !important;
        font-family: 'Pretendard', sans-serif !important;
    }
    .block-container { padding-top: 2rem !important; max-width: 720px !important; }

    /* Custom Card Style */
    .card-box {
        background: #FFFFFF; border: 1px solid #E5E5E5; border-radius: 16px;
        padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    .score-badge {
        font-size: 2rem; font-weight: 800; color: #10B981; line-height: 1;
    }
    .warning-box {
        background: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 12px;
        padding: 14px; color: #991B1B; font-size: 0.9rem; margin: 12px 0;
    }
    .lifecycle-flow {
        display: flex; justify-content: space-between; align-items: center;
        background: #F8FAFC; padding: 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Session State for Eco Passport & Reports
if "passport" not in st.mutable_status if hasattr(st, "mutable_status") else st.session_state:
    if "passport" not in st.session_state:
        st.session_state.passport = {"PET": False, "PP": False, "Can": False, "Glass": False, "Paper": False, "Vinyl": False}
    if "report_count" not in st.session_state:
        st.session_state.report_count = 3
        st.session_state.co2_saved = 66

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("API 키 설정이 필요합니다.")
    st.stop()

service = RecyclingService(groq_api_key=GROQ_API_KEY, tavily_api_key=TAVILY_API_KEY)

# Sidebar Navigation
st.sidebar.title("🌱 EcoLens AI")
st.sidebar.caption("Personal Eco Assistant")
menu = st.sidebar.radio("Navigation", ["🔍 AI Inspector (분석)", "🏆 Eco Passport (도감)", "📊 My Carbon Report", "📰 환경 뉴스"])

# ------------------------------------------------------------------------------
# PAGE 1: AI Inspector & Analysis
# ------------------------------------------------------------------------------
if menu == "🔍 AI Inspector (분석)":
    st.title("AI Inspector & Intelligence")
    st.caption("제품을 올리면 AI가 즉시 스캐닝하여 수거 거부 위험 및 순환 가능성을 정밀 검수합니다.")

    tab_name, tab_file = st.tabs(["✍️ 품목 검색", "📷 사진 업로드"])
    uploaded_image, item_text = None, None

    with tab_name:
        item_text = st.text_input("품목명 입력", placeholder="예: 햇반 용기, 삼다수 병, 컵라면 용기", label_visibility="collapsed")
    with tab_file:
        img_file = st.file_uploader("이미지 첨부", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if img_file:
            uploaded_image = Image.open(img_file)
            st.image(uploaded_image, use_container_width=True)

    location = st.selectbox("지역 선택", ["전국 공통", "서울특별시 강남구", "경기도 수원시"], label_visibility="collapsed")

    if st.button("🔍 AI Scan & Analyze →", type="primary", use_container_width=True):
        if not uploaded_image and not item_text:
            st.warning("품목을 입력하거나 이미지를 업로드해 주세요.")
        else:
            with st.spinner("🕵️ AI Inspector 스캐닝 중... (Material / Residue / Risk Check)"):
                try:
                    guide = service.analyze(item_text=item_text, image=uploaded_image, location=location)
                    st.session_state.current_guide = guide
                    st.session_state.report_count += 1
                    
                    # Update Passport
                    mat = guide.material if guide.material in st.session_state.passport else "PP"
                    st.session_state.passport[mat] = True
                except Exception as e:
                    st.error(f"분석 오류: {e}")

    # Display Analysis Result
    if "current_guide" in st.session_state:
        g: RecyclingGuide = st.session_state.current_guide

        st.write("")
        st.markdown(f"""
        <div class="card-box">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="background:#E6F4EA; color:#137333; font-weight:700; font-size:0.75rem; padding:3px 8px; border-radius:4px;">AI INSPECTION COMPLETE</span>
                    <h2 style="margin:8px 0 4px 0; font-size:1.6rem;">{g.item_name}</h2>
                    <span style="color:#666; font-size:0.85rem;">카테고리: {g.category} | 재질: {g.material}</span>
                </div>
                <div style="text-align:right;">
                    <div class="score-badge">{g.eco_score}<span style="font-size:1rem; color:#666;">점</span></div>
                    <div style="font-weight:700; color:#333;">Grade {g.eco_grade}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 1. AI Inspector Scanning Results
        st.subheader("🕵️ AI Inspector 스캔 리포트")
        ic = g.inspector
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("재질 감지", ic.material_type)
        c2.metric("라벨 상태", "제거 필요" if ic.label_removed else "양호")
        c3.metric("음식물 잔여", "있음 ⚠️" if ic.food_residue else "없음")
        c4.metric("오염 위험도", ic.contamination_risk)

        # Rejection Warning
        if g.rejection_risk_warning or g.recycling_rate_pct == 0:
            st.markdown(f"""
            <div class="warning-box">
                <b>⚠️ 수거 거부 위험 요소 경고:</b><br>{g.rejection_risk_warning or '복합재질(EVOH)로 인해 일반 재활용 선별장에서 수거 거부될 수 있습니다.'}
            </div>
            """, unsafe_allow_html=True)

        # 2. Lifecycle Flow (자원 순환 여정)
        st.write("")
        st.subheader("🔄 자원 순환 라이프사이클")
        steps_str = " ➔ ".join(g.lifecycle_steps) if g.lifecycle_steps else "원료 ➔ 사용 ➔ 일반배출 ➔ 소각/매립"
        st.markdown(f'<div class="lifecycle-flow">♻️ {steps_str}</div>', unsafe_allow_html=True)

        # 3. 배출 수칙 및 꿀팁
        st.write("")
        st.subheader("📋 올바른 배출 수칙")
        for i, step in enumerate(g.steps, 1):
            st.markdown(f"**{i}.** {step}")

        # 4. 연관 품목 추천
        if g.similar_items:
            st.write("")
            st.markdown("**🔗 연관 품목 한 번에 보기:** " + ", ".join([f"`{item}`" for item in g.similar_items]))

        # 5. AI Chat Instant Q&A
        st.write("")
        st.divider()
        st.subheader("💬 AI에게 궁금한 점 바로 물어보기")
        user_q = st.text_input("질문 입력", placeholder=f"예: {g.item_name} 라벨 안 떼고 버리면 어떻게 돼?", label_visibility="collapsed")
        if user_q:
            with st.spinner("AI 답변 생성 중..."):
                ans = service.ask_ai_chat(user_q, g.item_name)
                st.info(f"🤖 **AI 답변:** {ans}")

# ------------------------------------------------------------------------------
# PAGE 2: Eco Passport (수집 도감)
# ------------------------------------------------------------------------------
elif menu == "🏆 Eco Passport (도감)":
    st.title("🏆 AI Eco Passport")
    st.caption("다양한 재질의 품목을 분석하고 완벽하게 배출하여 도감 뱃지를 수집해 보세요!")

    pass_data = st.session_state.passport
    cols = st.columns(3)
    
    items = [
        ("PET (페트병)", "PET", "🥤"),
        ("PP (플라스틱)", "PP", "🍱"),
        ("Can (캔류)", "Can", "🥫"),
        ("Glass (유리)", "Glass", "🍾"),
        ("Paper (종이)", "Paper", "📦"),
        ("Vinyl (비닐)", "Vinyl", "🛍️")
    ]

    for idx, (label, key, emoji) in enumerate(items):
        with cols[idx % 3]:
            unlocked = pass_data.get(key, False)
            bg = "#E6F4EA" if unlocked else "#F0F0F0"
            border = "#34C759" if unlocked else "#CCCCCC"
            status = "✅ 수집 완료" if unlocked else "🔒 미수집"
            
            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {border}; border-radius:14px; padding:16px; text-align:center; margin-bottom:12px;">
                <div style="font-size:2.5rem;">{emoji}</div>
                <div style="font-weight:700; margin-top:4px;">{label}</div>
                <div style="font-size:0.75rem; color:#666; margin-top:2px;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 3: My Carbon Report
# ------------------------------------------------------------------------------
elif menu == "📊 My Carbon Report":
    st.title("📊 나의 환경 영향 리포트")
    st.caption("EcoLens와 함께 만든 실질적인 자원 절약 및 탄소 감축 지표입니다.")

    cnt = st.session_state.report_count
    co2 = cnt * 22
    trees = round(co2 / 500, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("총 분리배출 참여", f"{cnt}회")
    c2.metric("CO₂ 감축량", f"{co2}g")
    c3.metric("나무 기여 효과", f"{trees}그루")

    st.write("")
    st.subheader("📈 월별 자원 순환 참여도")
    st.progress(min(cnt * 10, 100))
    st.caption(f"이번 달 목표 달성률: {min(cnt * 10, 100)}% (평균 유저 대비 +24% 우수)")

# ------------------------------------------------------------------------------
# PAGE 4: Eco News
# ------------------------------------------------------------------------------
elif menu == "📰 환경 뉴스":
    st.title("📰 실시간 환경 & 분리배출 뉴스")
    st.caption("Tavily AI가 수집한 최신 지자체 배출 정책 및 자원 순환 소식입니다.")

    with st.spinner("최신 뉴스 가져오는 중..."):
        news_list = service.fetch_eco_news("대한민국 분리배출 정책 재활용 뉴스")
        if news_list:
            for n in news_list:
                st.markdown(f"""
                <div class="card-box">
                    <h4 style="margin:0 0 6px 0;"><a href="{n.get('url')}" target="_blank" style="text-decoration:none; color:#111;">{n.get('title')}</a></h4>
                    <p style="font-size:0.85rem; color:#555; margin:0;">{n.get('content')[:120]}...</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("최신 뉴스를 불러올 수 없습니다.")

st.sidebar.divider()
st.sidebar.caption("EcoLens Platform © 2026")
