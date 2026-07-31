import os
import json
import base64
import random
from datetime import datetime
from typing import List, Optional
from PIL import Image
import streamlit as st
from pydantic import BaseModel, Field
from groq import Groq
from tavily import TavilyClient

# ==============================================================================
# 1. Pydantic 데이터 모델 정의
# ==============================================================================

class SourceItem(BaseModel):
    title: str = Field(description="출처 기관명 또는 문서 제목")
    url: str = Field(description="원문 URL")
    published_or_checked: str = Field(description="확인일자 또는 작성일자")

class RecyclingGuide(BaseModel):
    item_name: str = Field(description="판별된 정확한 품목명")
    material: str = Field(description="추정 재질 (예: PET, PP, 종이팩, 유리, 복합재질 등)")
    category: str = Field(description="대분류 (예: 플라스틱, 비닐류, 일반쓰레기, 대형폐기물, 폐가전 등)")
    confidence: str = Field(description="판별 신뢰도 ('high', 'medium', 'low')")
    steps: List[str] = Field(description="실행형 3~4단계 배출 절차", min_items=1, max_items=4)
    cautions: List[str] = Field(description="배출 시 주의사항 및 예외 조건")
    local_note: Optional[str] = Field(default=None, description="지역별 전용 수거 기준 관련 참고사항")
    sources: List[SourceItem] = Field(default=[], description="검색된 근거 정보 출처 목록")
    eco_tip: str = Field(description="친환경 실천 팁 및 실천 행동")
    co2_reduction_est: str = Field(description="올바른 분리배출 시 예상 효과 (예: '약 15g CO2 절감 효과')")
    recycling_grade: str = Field(description="재활용 용이성 등급 ('최우수', '우수', '보통', '어려움')")
    needs_confirmation: bool = Field(default=False, description="지자체/상세 재질 추가 확인 필요 여부")


# ==============================================================================
# 2. Groq & Tavily RAG 서비스 클래스
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
                max_results=3,
                include_domains=["go.kr", "or.kr", "seoul.go.kr", "keco.or.kr"]
            )
            results = response.get("results", [])
            if not results:
                response = self.tavily_client.search(query=query, search_depth="basic", max_results=3)
                results = response.get("results", [])
            return results
        except Exception as e:
            st.warning(f"Tavily 검색 연결 실패 (기본 규정으로 처리): {e}")
            return []

    def encode_image_to_base64(self, image: Image.Image) -> str:
        import io
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_and_guide(
        self,
        item_text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        location: str = "전국 공통"
    ) -> RecyclingGuide:
        
        system_instruction = f"""
        당신은 대한민국 환경부 및 지자체 분리배출 기준에 정통한 친환경 AI 도우미 'EcoLens AI'입니다.
        사용자의 입력(이미지 및 텍스트)을 분석하여 올바른 분리배출 가이드를 반드시 주어진 JSON 스키마 형식에 맞춰 제공하세요.
        
        [지침]
        1. 이미지나 텍스트가 불명확하면 confidence를 'low'로 설정하고 needs_confirmation을 true로 설정하세요.
        2. 검색 컨텍스트에 나와있지 않은 지자체 특화 규정은 절대 지어내지 말고, 불확실하면 지자체 확인 필요라고 안내하세요.
        3. steps는 사용자가 바로 행동할 수 있는 3~4단계 절차로 작성하세요.
        4. recycling_grade는 '최우수', '우수', '보통', '어려움' 중 하나로 정하세요.
        
        [반환할 JSON 구조]
        {json.dumps(RecyclingGuide.model_json_schema(), ensure_ascii=False, indent=2)}
        """

        search_query = f"{location} {item_text or '쓰레기'} 분리배출 기준 2026"
        search_results = self.search_local_rules(search_query)
        context_str = "\n".join([f"- 제목: {r['title']}\n  URL: {r['url']}\n  내용: {r['content']}" for r in search_results])

        user_content = [
            {
                "type": "text",
                "text": f"""[사용자 입력]
지역: {location}
물품 설명: {item_text or '사진 참고'}

[검색된 최신 지자체/공공 정보]
{context_str if context_str else "검색 결과 없음. 기본 전국 공통 기준 적용."}

위 정보를 바탕으로 스키마 형태의 JSON 응답만 출력하세요."""
            }
        ]

        if image:
            base64_image = self.encode_image_to_base64(image)
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            model_name = "llama-3.2-11b-vision-preview"
        else:
            model_name = "llama-3.3-70b-versatile"

        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        json_response = response.choices[0].message.content
        return RecyclingGuide.model_validate_json(json_response)


# ==============================================================================
# 3. Streamlit UI & 세션 상태 관리
# ==============================================================================

st.set_page_config(
    page_title="EcoLens AI — 스마트 AI 분리배출 센터",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []
if "action_count" not in st.session_state:
    st.session_state.action_count = 0
if "saved_co2" not in st.session_state:
    st.session_state.saved_co2 = 0.0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

# 강제 라이트 모드 및 고급 애니메이션 CSS
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    /* 1. 강제 라이트 모드 배경 고정 */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Pretendard', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }

    /* Keyframe 애니메이션 정의 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* 요소별 애니메이션 적용 */
    .animated-card {
        animation: fadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .pulse-badge {
        animation: pulseGlow 2s infinite;
    }

    /* UI 컴포넌트 라이트 모드 패치 */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }

    /* Custom Header */
    .header-banner {
        background: linear-gradient(135deg, #059669 0%, #10B981 50%, #34D399 100%);
        padding: 32px 40px;
        border-radius: 24px;
        color: white !important;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
    }
    
    .header-banner h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        margin: 0;
        font-size: 2.3rem;
    }

    /* Stat Dashboard Cards */
    .stat-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #059669;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Card Box UI */
    .card-box {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }

    /* Badges */
    .badge-grade {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 12px;
        font-size: 0.85rem;
    }

    .step-box {
        background: #F8FAFC;
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# API 키 가져오기
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")

# 상단 헤더 배너
st.markdown("""
<div class="header-banner animated-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>🌱 EcoLens AI 센터</h1>
            <p style="margin-top: 8px; opacity: 0.95; font-size: 1.05rem;">
                Groq Llama 3.3 초고속 추론 엔진 & RAG 기반의 실시간 지자체 분리배출 플랫폼
            </p>
        </div>
        <div class="pulse-badge" style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 30px; font-weight: 700;">
            ⚡ LIVE ENGINE ACTIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("⚠️ API 키가 설정되지 않았습니다. `.streamlit/secrets.toml`에 `GROQ_API_KEY`와 `TAVILY_API_KEY`를 추가해 주세요.")
    st.stop()

service = RecyclingService(groq_api_key=GROQ_API_KEY, tavily_api_key=TAVILY_API_KEY)

# 1. 상단 대시보드 (4 컬럼 구성)
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.markdown(f"""
    <div class="stat-card animated-card">
        <div class="stat-value">{len(st.session_state.history)}건</div>
        <div class="stat-label">🔍 누적 판별 횟수</div>
    </div>
    """, unsafe_allow_html=True)
with d2:
    st.markdown(f"""
    <div class="stat-card animated-card">
        <div class="stat-value">{st.session_state.action_count}회</div>
        <div class="stat-label">✅ 올바른 배출 실천</div>
    </div>
    """, unsafe_allow_html=True)
with d3:
    st.markdown(f"""
    <div class="stat-card animated-card">
        <div class="stat-value" style="color: #2563EB;">{st.session_state.saved_co2:.1f}g</div>
        <div class="stat-label">🍃 절감한 예상 CO2</div>
    </div>
    """, unsafe_allow_html=True)
with d4:
    st.markdown(f"""
    <div class="stat-card animated-card">
        <div class="stat-value" style="color: #D97706;">{st.session_state.quiz_score}점</div>
        <div class="stat-label">🏆 환경 퀴즈 포인트</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 2. 메인 3컬럼 레이아웃 (좌: 입력 / 중: 결과 리포트 / 우: 보조 기능 및 퀴즈)
col_left, col_mid, col_right = st.columns([1.1, 1.3, 1], gap="medium")

# ------------------------------------------------------------------------------
# [왼쪽 컬럼] 입력 및 조건 설정
# ------------------------------------------------------------------------------
with col_left:
    st.markdown('<div class="card-box animated-card">', unsafe_allow_html=True)
    st.subheader("🔍 품목 판별하기")
    
    tab_img, tab_txt = st.tabs(["📷 사진 첨부", "✍️ 텍스트 검색"])
    
    uploaded_image = None
    item_text = None
    
    with tab_img:
        img_file = st.file_uploader("배출할 물품 사진 선택", type=["jpg", "jpeg", "png", "webp"])
        if img_file:
            uploaded_image = Image.open(img_file)
            st.image(uploaded_image, caption="업로드된 영상/이미지", use_container_width=True)
            
    with tab_txt:
        item_text = st.text_input("품목명 직접 입력", placeholder="예: 햇반 용기, 컵라면 용기, 뽁뽁이")

    location = st.selectbox(
        "📍 배출 지역 선택",
        ["전국 공통", "서울특별시 강남구", "서울특별시 마포구", "경기도 수원시", "부산광역시 해운대구", "대구광역시 수성구", "인천광역시 연수구"],
        index=0
    )

    analyze_btn = st.button("✨ 초고속 AI 분석 시작", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 지자체 수거요일 정보 가이드 (화면을 꽉 채우기 위한 추가 정보 블록)
    st.markdown("""
    <div class="card-box animated-card" style="margin-top: 15px;">
        <h4 style="margin-top:0; color:#0F172A;">📅 일반적인 지자체 수거 원칙</h4>
        <ul style="font-size:0.88rem; color:#475569; padding-left:20px; margin-bottom:0;">
            <li><b>투명 페트병:</b> 지정된 요일 별도 배출 (라벨 제거 필수)</li>
            <li><b>음식물 용기:</b> 이물질 완벽 세척 후 플라스틱 배출</li>
            <li><b>스티로폼:</b> 흰색 단일 소재만 가능 (택배 운송장 제거)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# [중앙 컬럼] AI 리포트 결과 출력
# ------------------------------------------------------------------------------
with col_mid:
    st.markdown('<div class="card-box animated-card">', unsafe_allow_html=True)
    st.subheader("📋 AI 스마트 분석 리포트")
    
    if analyze_btn:
        if not uploaded_image and not item_text:
            st.warning("사진을 업로드하거나 품목명을 입력해 주세요.")
        else:
            with st.spinner("⚡ Groq Llama 3.3이 RAG 정보와 함께 분석 중입니다..."):
                try:
                    guide = service.analyze_and_guide(
                        item_text=item_text,
                        image=uploaded_image,
                        location=location
                    )

                    # 세션 히스토리 및 통계 업데이트
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M"),
                        "name": guide.item_name,
                        "category": guide.category,
                    })
                    st.session_state.saved_co2 += 15.0  # 평균 절감치 추가

                    st.markdown(f"""
                    <div style="background:#F1F5F9; padding:18px; border-radius:14px; margin-bottom:15px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="badge-grade">재활용 등급: {guide.recycling_grade}</span>
                            <span style="font-size:0.85rem; color:#64748B;">신뢰도: <b>{guide.confidence.upper()}</b></span>
                        </div>
                        <h2 style="margin: 10px 0 0 0; color:#0F172A;">{guide.item_name}</h2>
                        <p style="margin:4px 0 0 0; color:#475569; font-size:0.95rem;">
                            재질: <b>{guide.material}</b> | 분류: <b>{guide.category}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if guide.needs_confirmation:
                        st.info("💡 용기 하단 표기나 이물질 묻음 여부를 한번 더 확인해 주세요.")

                    st.markdown("#### 🛠️ 단계별 배출 가이드")
                    for idx, step in enumerate(guide.steps, 1):
                        st.markdown(f"""
                        <div class="step-box">
                            <b style="color:#10B981;">0{idx}.</b> {step}
                        </div>
                        """, unsafe_allow_html=True)

                    if guide.cautions:
                        st.markdown("#### ⚠️ 주의사항")
                        for caution in guide.cautions:
                            st.error(f"• {caution}")

                    if guide.local_note:
                        st.warning(f"📍 **{location} 안내:** {guide.local_note}")

                    st.markdown(f"""
                    <div style="background:#ECFDF5; border:1px solid #A7F3D0; padding:16px; border-radius:12px; margin-top:15px;">
                        <b style="color:#047857;">🌱 친환경 실천 팁:</b> {guide.eco_tip}<br>
                        <small style="color:#065F46;"> 예상 효과: {guide.co2_reduction_est}</small>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("")
                    if st.button("✅ 배출 완료 인증 및 CO2 절감하기", use_container_width=True):
                        st.session_state.action_count += 1
                        st.balloons()
                        st.success("실천 인증이 완료되어 CO2 절감량이 반영되었습니다!")

                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")
    else:
        st.info("👈 좌측에서 분석을 시작하면 상세 리포트가 이곳에 나타납니다.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# [오른쪽 컬럼] 미니 게임, 히스토리 및 Q&A (화면을 가득 채우는 서브 모듈)
# ------------------------------------------------------------------------------
with col_right:
    # 1. 최근 분석 기록
    st.markdown('<div class="card-box animated-card">', unsafe_allow_html=True)
    st.subheader("📜 최근 판별 기록")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-4:]):
            st.markdown(f"• **{item['name']}** <small>({item['category']}) - {item['time']}</small>", unsafe_allow_html=True)
    else:
        st.caption("아직 기록이 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. 미니 분리배출 퀴즈 (참여형 요소)
    st.markdown('<div class="card-box animated-card">', unsafe_allow_html=True)
    st.subheader("🧩 1초 환경 퀴즈")
    st.write("**Q. 깨진 유리는 유리 재활용으로 배출해야 할까요?**")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if st.button("⭕ 그렇다"):
            st.error("❌ 틀렸습니다! 깨진 유리는 쓰레기 종량제 봉투나 불연성 마대에 버려야 합니다.")
    with q_col2:
        if st.button("❌ 아니다"):
            st.success("⭕ 정답입니다! 깨진 유리는 재활용이 불가능합니다.")
            st.session_state.quiz_score += 10
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. 자주 묻는 질문 (FAQ)
    st.markdown('<div class="card-box animated-card">', unsafe_allow_html=True)
    st.subheader("❓ 자주 묻는 Q&A")
    with st.expander("Q. 씻어도 안 지워지는 용기는?"):
        st.write("양념이 완전히 착색된 스티로폼이나 플라스틱은 일반 쓰레기로 버려야 합니다.")
    with st.expander("Q. 영수증은 종이류인가요?"):
        st.write("혼합 재질(감열지)이므로 일반 쓰레기로 배출해야 합니다.")
    st.markdown('</div>', unsafe_allow_html=True)
