import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
import streamlit as st
from pydantic import BaseModel, Field
from groq import Groq

# ==============================================================================
# 1. Enterprise Inspection Schema
# ==============================================================================
class SubComponent(BaseModel):
    name: str = Field(description="부품명 (예: 플라스틱 용기, 비닐 필름, 종이 띠지)")
    material: str = Field(description="재질 (PP, PET, Paper, Vinyl, Other)")
    recycling_rate: int = Field(description="재활용률 (%)")
    status: str = Field(description="검사 상태 (PASS, WARNING, FAIL)")

class InspectionChecklist(BaseModel):
    object_detection: str = Field(description="객체 인식 (PASS, WARNING, FAIL)")
    material_analysis: str = Field(description="재질 정밀 감지 (PASS, WARNING, FAIL)")
    residue_detection: str = Field(description="잔여물 검수 (PASS, WARNING, FAIL)")
    label_detection: str = Field(description="라벨/필름 분리 검수 (PASS, WARNING, FAIL)")
    cap_detection: str = Field(description="뚜껑/이물질 검수 (PASS, WARNING, FAIL)")
    local_policy: str = Field(description="지자체 정책 준수 (PASS, WARNING, FAIL)")

class ConfidenceMetrics(BaseModel):
    material_confidence: int = Field(description="재질 인식 신뢰도 (%)")
    category_confidence: int = Field(description="분류 신뢰도 (%)")
    risk_confidence: int = Field(description="위험 감지 신뢰도 (%)")

class ScoreBreakdown(BaseModel):
    material_score: int = Field(description="재질 평가 점수 (max 25)")
    cleaning_score: int = Field(description="세척 상태 점수 (max 20)")
    contamination_score: int = Field(description="오염도 점수 (max 30)")
    policy_score: int = Field(description="규정 적합 점수 (max 25)")

class InspectionReport(BaseModel):
    detected_item: str = Field(description="감지된 품목 대표명 (예: 도시락, 햇반)")
    is_ambiguous: bool = Field(description="복합 제품 여부")
    components: List[SubComponent] = Field(description="구성 부품 전체 리스트")
    default_component_index: int = Field(description="기본 선택 부품 인덱스 (0부터 시작)")
    
    total_score: int = Field(description="총점 (0~100)")
    grade: str = Field(description="AAA, AA, A, B, C 중 선택")
    score_breakdown: ScoreBreakdown = Field(description="세부 점수")
    checklist: InspectionChecklist = Field(description="6대 검사 스탬프")
    confidence: ConfidenceMetrics = Field(description="신뢰도 히트맵")
    
    primary_category: str = Field(description="배출 카테고리 (예: 플라스틱류 / 일반쓰레기)")
    steps: List[str] = Field(description="실행 수칙")
    warning_notes: Optional[str] = Field(description="주의 경고")

# ==============================================================================
# 2. EcoLens Intelligence Engine
# ==============================================================================
class EcoLensEngine:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)

    def run_single_pass_inspection(self, item_text: str, location: str) -> InspectionReport:
        system_prompt = f"""
        당신은 산업용 자원순환 AI 검수 엔진 'EcoLens Intelligence Engine'입니다.
        단 한 번의 스캔으로 입력된 품목의 복합 구성 요소, 6대 검사 스탬프(PASS/WARNING/FAIL), 세부 점수(100점 만점) 및 신뢰도를 정밀 산출하세요.
        도시락, 햇반, 컵라면 같은 복합 제품은 반드시 is_ambiguous=True로 처리하고 용기, 필름, 뚜껑 등 components를 분리하세요.
        다음 JSON 스키마를 엄격히 준수하여 반환하세요:
        {json.dumps(InspectionReport.model_json_schema(), ensure_ascii=False, indent=2)}
        """
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"배출 지역: {location}\n검사 대상 입력: {item_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return InspectionReport.model_validate_json(response.choices[0].message.content)

# ==============================================================================
# 3. Canva-Style Modern Infographic CSS Setup
# ==============================================================================
st.set_page_config(page_title="EcoLens Infographic Report", page_icon="🌱", layout="centered")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #F1F5F9 !important;
        color: #0F172A !important;
        font-family: 'Pretendard', sans-serif !important;
    }
    .block-container { padding-top: 2rem !important; max-width: 780px !important; }

    /* Canva Poster Card Shell */
    .canva-card {
        background: #FFFFFF;
        border-radius: 28px;
        box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.12);
        overflow: hidden;
        border: 1px solid #E2E8F0;
        margin-top: 15px;
    }

    /* Top Canva Header Banner */
    .canva-banner {
        background: linear-gradient(135deg, #059669 0%, #10B981 50%, #34D399 100%);
        padding: 30px 36px;
        color: #FFFFFF;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
    }
    .canva-banner-title {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: 0.9;
    }
    .canva-banner-item {
        font-size: 2rem;
        font-weight: 900;
        margin: 4px 0 0 0;
        line-height: 1.1;
    }

    /* Canva Body Container */
    .canva-body {
        padding: 32px 36px;
    }

    /* Score Badge Floating Circle */
    .score-badge {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 12px 24px;
        text-align: center;
    }
    .score-num { font-size: 2.6rem; font-weight: 900; line-height: 1; color: #FFFFFF; }
    .score-grade { font-size: 0.85rem; font-weight: 800; opacity: 0.95; }

    /* Section Grid */
    .canva-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        margin-top: 20px;
    }

    /* Inner Cards */
    .inner-box {
        background: #F8FAFC;
        border-radius: 18px;
        padding: 20px;
        border: 1px solid #E2E8F0;
    }
    .inner-box-title {
        font-size: 0.85rem;
        font-weight: 800;
        color: #64748B;
        margin-bottom: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Progress & Bar Customization */
    .metric-row {
        margin-bottom: 12px;
    }
    .metric-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 4px;
    }
    .bar-bg {
        background: #E2E8F0;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #10B981, #059669);
        border-radius: 4px;
    }

    /* Stamps */
    .c-stamp {
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 800;
        display: inline-block;
    }
    .c-pass { background: #DCFCE7; color: #15803D; }
    .c-warn { background: #FEF9C3; color: #A16207; }
    .c-fail { background: #FEE2E2; color: #B91C1C; }

    /* Checklist Canva List */
    .c-check-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px dashed #CBD5E1;
        font-size: 0.82rem;
        font-weight: 600;
        color: #1E293B;
    }
    .c-check-item:last-child { border-bottom: none; }

    /* Radio UI Customization */
    div[data-testid="stRadio"] label p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }
</style>
""", unsafe_allow_html=True)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY 설정이 필요합니다.")
    st.stop()

engine = EcoLensEngine(groq_api_key=GROQ_API_KEY)

# Top Bar
st.title("🌱 EcoLens Infographic Generator")

# Inputs
target_input = st.text_input("검사 대상 입력", placeholder="예: 도시락, 햇반, 삼다수, 컵라면", key="canva_input")
location = st.selectbox("배출 규정 지역", ["전국 공통 기준", "서울특별시 강남구", "경기도 수원시"], label_visibility="collapsed")

if st.button("✨ 캔바 스타일 보고서 생성하기", type="primary", use_container_width=True):
    if target_input:
        with st.spinner("디자인 보고서 카드 제작 중..."):
            try:
                report = engine.run_single_pass_inspection(target_input, location)
                st.session_state.report = report
                st.session_state.selected_comp_idx = report.default_component_index
                st.session_state.doc_id = f"CANVA-{uuid.uuid4().hex[:6].upper()}"
                st.session_state.created_at = datetime.now().strftime("%Y.%m.%d")
            except Exception as e:
                st.error(f"생성 실패: {e}")

# ==============================================================================
# CANVA INFOGRAPHIC CARD RENDER
# ==============================================================================
if "report" in st.session_state and st.session_state.report:
    rep: InspectionReport = st.session_state.report
    doc_id = st.session_state.get("doc_id", "CANVA-001")
    created_at = st.session_state.get("created_at", "2026.04.04")

    # Sub-component Switcher
    if rep.is_ambiguous and rep.components:
        st.write("")
        st.caption("💡 구성품 선택:")
        comp_names = [f"{c.name} ({c.material})" for c in rep.components]
        selected_tab = st.radio("구성 요소:", comp_names, horizontal=True, label_visibility="collapsed")
        st.session_state.selected_comp_idx = comp_names.index(selected_tab)

    active_comp = rep.components[st.session_state.selected_comp_idx] if rep.components else None

    def get_c_stamp(status: str):
        if status == "PASS": return '<span class="c-stamp c-pass">PASS</span>'
        elif status == "WARNING": return '<span class="c-stamp c-warn">WARNING</span>'
        return '<span class="c-stamp c-fail">FAIL</span>'

    sb = rep.score_breakdown
    chk = rep.checklist

    # Canva Infographic HTML Output
    canva_html = f"""
    <div class="canva-card">
        <!-- Banner Header -->
        <div class="canva-banner">
            <div>
                <div class="canva-banner-title">ECOLENS RECYCLING REPORT • {doc_id}</div>
                <div class="canva-banner-item">{rep.detected_item}</div>
                <div style="font-size:0.9rem; opacity:0.9; margin-top:4px;">
                    분류: <b>{rep.primary_category}</b> {f'| 세부: {active_comp.name}' if active_comp else ''}
                </div>
            </div>
            <div class="score-badge">
                <div class="score-num">{rep.total_score}</div>
                <div class="score-grade">GRADE {rep.grade}</div>
            </div>
        </div>

        <!-- Body Grid -->
        <div class="canva-body">
            <div style="display:flex; justify-content:space-between; color:#64748B; font-size:0.75rem; font-weight:700; margin-bottom:12px;">
                <span>ISSUED BY ECOLENS AI</span>
                <span>DATE: {created_at}</span>
            </div>

            <div class="canva-grid">
                <!-- Left Box: Score Bars -->
                <div class="inner-box">
                    <div class="inner-box-title">📊 세부 진단 점수</div>
                    
                    <div class="metric-row">
                        <div class="metric-label"><span>재질 평가</span><span>{sb.material_score} / 25</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{(sb.material_score/25)*100}%;"></div></div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-label"><span>세척 상태</span><span>{sb.cleaning_score} / 20</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{(sb.cleaning_score/20)*100}%;"></div></div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-label"><span>오염도 점수</span><span>{sb.contamination_score} / 30</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{(sb.contamination_score/30)*100}%;"></div></div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-label"><span>규정 적합성</span><span>{sb.policy_score} / 25</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{(sb.policy_score/25)*100}%;"></div></div>
                    </div>
                </div>

                <!-- Right Box: Inspection Checklist -->
                <div class="inner-box">
                    <div class="inner-box-title">🔍 6대 정밀 검사 스탬프</div>
                    <div class="c-check-item"><span>객체 인식</span>{get_c_stamp(chk.object_detection)}</div>
                    <div class="c-check-item"><span>재질 분석</span>{get_c_stamp(chk.material_analysis)}</div>
                    <div class="c-check-item"><span>잔여물 검수</span>{get_c_stamp(chk.residue_detection)}</div>
                    <div class="c-check-item"><span>라벨 분리</span>{get_c_stamp(chk.label_detection)}</div>
                    <div class="c-check-item"><span>뚜껑/이물질</span>{get_c_stamp(chk.cap_detection)}</div>
                    <div class="c-check-item"><span>지자체 규정</span>{get_c_stamp(chk.local_policy)}</div>
                </div>
            </div>

            <!-- Steps Footer Card -->
            <div class="inner-box" style="margin-top:20px; background:#F1F5F9;">
                <div class="inner-box-title">📋 올바른 분리배출 가이드</div>
                <div style="font-size:0.85rem; color:#334155; line-height:1.6;">
                    {'<br>'.join([f"<b>{i+1}.</b> {step}" for i, step in enumerate(rep.steps)])}
                </div>
                {f'<div style="margin-top:10px; color:#991B1B; font-size:0.8rem; font-weight:700;">⚠️ {rep.warning_notes}</div>' if rep.warning_notes else ''}
            </div>
        </div>
    </div>
    """

    st.markdown(canva_html, unsafe_allow_html=True)

st.divider()
st.caption("EcoLens Infographic Generator © 2026")
