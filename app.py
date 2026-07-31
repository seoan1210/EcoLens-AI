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

    def ask_intelligence_fallback(self, question: str, report: InspectionReport) -> str:
        prompt = f"""
        [Inspection Report Context]
        - 품목: {report.detected_item}
        - 등급: {report.grade} ({report.total_score}점)
        - 주요 수칙: {', '.join(report.steps)}
        - 경고: {report.warning_notes or '없음'}

        사용자 질문: "{question}"
        위 Report 데이터에 입각해서만 2문장 이내로 명확히 답하세요. 모순되는 답변은 금지합니다.
        """
        res = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content

# ==============================================================================
# 3. Streamlit Liquid Glass & Report Design Setup
# ==============================================================================
st.set_page_config(page_title="EcoLens Intelligence Report", page_icon="🌱", layout="centered")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%) !important;
        color: #0F172A !important;
        font-family: 'Pretendard', sans-serif !important;
    }
    .block-container { padding-top: 1.8rem !important; max-width: 760px !important; }

    /* Glass Engine Header */
    .engine-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 16px 22px; border-radius: 16px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 15px rgba(0,0,0,0.03); margin-bottom: 20px;
    }
    .engine-title { font-size: 1.15rem; font-weight: 800; color: #0F172A; }

    /* High-End Official Report Paper Box */
    .report-paper {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 1);
        padding: 32px;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.06);
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        margin-top: 15px; position: relative; overflow: hidden;
    }

    /* Document Watermark Accent */
    .report-paper::before {
        content: "ECOLENS CERTIFIED";
        position: absolute; right: -20px; bottom: 10px;
        font-size: 3.5rem; font-weight: 900;
        color: rgba(15, 23, 42, 0.02);
        letter-spacing: 4px; pointer-events: none; transform: rotate(-5deg);
    }

    /* Report Meta Header */
    .report-meta-bar {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 2px solid #0F172A; padding-bottom: 12px; margin-bottom: 20px;
    }
    .report-doc-id { font-size: 0.75rem; font-family: monospace; color: #64748B; font-weight: 600; }

    /* Score breakdown inner glass grid */
    .score-grid {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
        text-align: center; background: rgba(241, 245, 249, 0.6);
        backdrop-filter: blur(8px); padding: 12px; border-radius: 12px;
        border: 1px solid rgba(226, 232, 240, 0.8); margin-top: 16px;
    }
    .score-grid div span { display: block; color: #64748B; font-size: 0.72rem; margin-bottom: 2px; }
    .score-grid div b { color: #0F172A; font-size: 0.95rem; }

    /* Stamps */
    .stamp-pass { background: #DCFCE7; color: #166534; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; border: 1px solid #BBF7D0; }
    .stamp-warning { background: #FEF9C3; color: #854D0E; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; border: 1px solid #FEF08A; }
    .stamp-fail { background: #FEE2E2; color: #991B1B; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; border: 1px solid #FCA5A5; }

    /* Checklist Item Box */
    .checklist-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 12px; margin-bottom: 6px; border-radius: 10px;
        background: rgba(255, 255, 255, 0.5); border: 1px solid rgba(226, 232, 240, 0.8);
        font-size: 0.88rem; color: #0F172A; font-weight: 500;
        transition: all 0.2s ease;
    }
    .checklist-row:hover { background: rgba(255, 255, 255, 0.9); transform: translateX(3px); }

    /* Streamlit Radio Override */
    div[data-testid="stRadio"] label p {
        color: #0F172A !important; font-weight: 700 !important; font-size: 0.88rem !important;
    }

    /* Print Optimization CSS */
    @media print {
        .block-container { max-width: 100% !important; padding: 0 !important; }
        .stButton, .stSelectbox, .stTextInput, .engine-header, [data-testid="stSidebar"] { display: none !important; }
        .report-paper { box-shadow: none !important; border: 1px solid #000 !important; background: #FFF !important; }
    }
</style>
""", unsafe_allow_html=True)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY 설정이 필요합니다.")
    st.stop()

engine = EcoLensEngine(groq_api_key=GROQ_API_KEY)

# Header
st.markdown("""
<div class="engine-header">
    <div class="engine-title">🌱 EcoLens Intelligence Engine</div>
    <span class="stamp-pass">ENTERPRISE REPORT GEN</span>
</div>
""", unsafe_allow_html=True)

# Input
target_input = st.text_input("검사 대상 입력", placeholder="예: 햇반, 도시락, 삼다수, 컵라면", key="engine_input_key")
location = st.selectbox("배출 규정 지역", ["전국 공통 기준", "서울특별시 강남구", "경기도 수원시"], label_visibility="collapsed")

if st.button("🔬 Execute Official Report →", type="primary", use_container_width=True):
    if target_input:
        with st.spinner("정밀 분석 보고서 생성 중..."):
            try:
                report = engine.run_single_pass_inspection(target_input, location)
                st.session_state.report = report
                st.session_state.selected_comp_idx = report.default_component_index
                st.session_state.doc_id = f"ECL-{uuid.uuid4().hex[:8].upper()}"
                st.session_state.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.active_chip_answer = None
            except Exception as e:
                st.error(f"보고서 생성 실패: {e}")

# ==============================================================================
# DISPLAY REPORT (Professional Sheet Format)
# ==============================================================================
if "report" in st.session_state and st.session_state.report:
    rep: InspectionReport = st.session_state.report
    doc_id = st.session_state.get("doc_id", "ECL-TEST-0001")
    created_at = st.session_state.get("created_at", "2026-04-04 12:00:00")
    
    st.write("")
    
    # Sub-component Tab Selector
    if rep.is_ambiguous and rep.components:
        st.caption("💡 복합 구성 품목이 감지되었습니다. 세부 부품 선택:")
        comp_names = [f"{c.name} ({c.material})" for c in rep.components]
        selected_tab = st.radio("구성 요소:", comp_names, horizontal=True, label_visibility="collapsed")
        st.session_state.selected_comp_idx = comp_names.index(selected_tab)

    active_comp = rep.components[st.session_state.selected_comp_idx] if rep.components else None

    def get_stamp_html(status: str):
        if status == "PASS": return '<span class="stamp-pass">PASS</span>'
        elif status == "WARNING": return '<span class="stamp-warning">WARNING</span>'
        return '<span class="stamp-fail">FAIL</span>'

    # Official Report Paper Layout
    report_html = f"""
    <div class="report-paper">
        <!-- Top Official Meta Header -->
        <div class="report-meta-bar">
            <div>
                <span style="font-size:0.75rem; font-weight:800; color:#059669; letter-spacing:1px;">OFFICIAL REPORT</span>
                <h3 style="margin:0; font-size:1.1rem; color:#0F172A;">자원순환 정밀 진단 보고서</h3>
            </div>
            <div style="text-align:right;">
                <div class="report-doc-id">DOC: {doc_id}</div>
                <div style="font-size:0.75rem; color:#64748B;">{created_at}</div>
            </div>
        </div>

        <!-- Main Info Grid -->
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
            <div>
                <span style="color:#64748B; font-size:0.78rem; font-weight:700;">INSPECTED TARGET</span>
                <h2 style="margin:2px 0 6px 0; font-size:1.6rem; color:#0F172A; font-weight:800;">
                    {rep.detected_item} {f' <span style="font-size:1.1rem; color:#475569;">({active_comp.name})</span>' if active_comp else ''}
                </h2>
                <span style="font-size:0.88rem; color:#334155;">권장 배출 분류: <b style="color:#0F172A;">{rep.primary_category}</b></span>
            </div>
            <div style="text-align:right; background:rgba(255,255,255,0.7); padding:10px 16px; border-radius:14px; border:1px solid #E2E8F0;">
                <div style="font-size:2.4rem; font-weight:900; color:#0F172A; line-height:1;">{rep.total_score}<span style="font-size:1rem; color:#64748B;">/100</span></div>
                <div style="font-weight:800; color:#166534; font-size:0.9rem; margin-top:3px;">Grade {rep.grade}</div>
            </div>
        </div>

        <!-- Score Grid Breakdown -->
        <div class="score-grid">
            <div><span>재질 평가</span><b>{rep.score_breakdown.material_score} / 25</b></div>
            <div><span>세척 상태</span><b>{rep.score_breakdown.cleaning_score} / 20</b></div>
            <div><span>오염도 점수</span><b>{rep.score_breakdown.contamination_score} / 30</b></div>
            <div><span>규정 적합성</span><b>{rep.score_breakdown.policy_score} / 25</b></div>
        </div>
    </div>
    """
    st.markdown(report_html, unsafe_allow_html=True)

    # Detailed Checklist Section
    st.write("")
    st.markdown("##### 🔍 6대 정밀 검사 항목 (Inspection Checklist)")
    
    chk = rep.checklist
    items_map = [
        ("Object Detection (객체 인식)", chk.object_detection),
        ("Material Analysis (재질 정밀 감지)", chk.material_analysis),
        ("Residue Detection (음식물/잔여물 검수)", chk.residue_detection),
        ("Label Detection (라벨/필름 분리 검수)", chk.label_detection),
        ("Cap Detection (뚜껑/이물질 검수)", chk.cap_detection),
        ("Local Policy (지자체 규정 적합성)", chk.local_policy),
    ]

    for label, status in items_map:
        st.markdown(f"""
        <div class="checklist-row">
            <span>{label}</span>
            <div>{get_stamp_html(status)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Confidence Progress
    st.write("")
    st.markdown("##### 📊 AI Confidence Metrics")
    col1, col2, col3 = st.columns(3)
    col1.caption(f"재질 인식: {rep.confidence.material_confidence}%")
    col1.progress(rep.confidence.material_confidence / 100)
    
    col2.caption(f"분류 정확도: {rep.confidence.category_confidence}%")
    col2.progress(rep.confidence.category_confidence / 100)
    
    col3.caption(f"위험 감지: {rep.confidence.risk_confidence}%")
    col3.progress(rep.confidence.risk_confidence / 100)

    # Action Steps & Alert Box
    st.write("")
    st.markdown("##### 📋 올바른 배출 실행 지침")
    for i, step in enumerate(rep.steps, 1):
        st.markdown(f"**{i}.** {step}")

    if rep.warning_notes:
        st.warning(f"⚠️ **Inspection Warning:** {rep.warning_notes}")

    # Export & Download Option
    st.divider()
    download_html = f"""
    <html>
        <head>
            <meta charset="utf-8">
            <title>{rep.detected_item} - EcoLens Report</title>
            <style>
                body {{ font-family: sans-serif; padding: 40px; background: #F8FAFC; color: #0F172A; }}
                .box {{ background: #FFF; padding: 30px; border-radius: 16px; border: 1px solid #CBD5E1; max-width: 650px; margin: 0 auto; }}
                .title {{ font-size: 24px; font-weight: bold; margin-bottom: 8px; }}
                .score {{ font-size: 32px; font-weight: bold; color: #059669; }}
                .badge {{ background: #E2E8F0; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="box">
                <div style="font-size:12px; color:#64748B;">EcoLens Inspection Report | Doc ID: {doc_id}</div>
                <div class="title">{rep.detected_item} 진단 리포트</div>
                <div>배출 분류: <b>{rep.primary_category}</b></div>
                <hr style="margin:20px 0; border:none; border-top:1px solid #E2E8F0;">
                <div>총점: <span class="score">{rep.total_score}점</span> (Grade {rep.grade})</div>
                <h4>배출 수칙</h4>
                <ol>
                    {''.join([f'<li>{s}</li>' for s in rep.steps])}
                </ol>
            </div>
        </body>
    </html>
    """

    col_dl1, col_dl2 = st.columns([2, 1])
    col_dl1.caption("📄 보고서를 파일로 소장하거나 브라우저(Ctrl+P)로 인쇄할 수 있습니다.")
    col_dl2.download_button(
        label="📥 HTML 보고서 저장",
        data=download_html,
        file_name=f"EcoLens_Report_{doc_id}.html",
        mime="text/html",
        use_container_width=True
    )

    # Q&A Interactive Section
    st.divider()
    st.markdown("### 💬 Ask Intelligence Engine")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    if col_a.button("🧼 세척 방법?", use_container_width=True):
        st.session_state.active_chip_answer = f"**세척 지침:** {rep.steps[0] if rep.steps else '물로 내용물을 깨끗이 헹군 후 배출하세요.'}"
    if col_b.button("🏷️ 라벨/필름?", use_container_width=True):
        st.session_state.active_chip_answer = f"**라벨 검수 상태 ({chk.label_detection}):** 비닐 라벨은 분리하여 별도 배출하세요."
    if col_c.button("❌ 일반쓰레기?", use_container_width=True):
        st.session_state.active_chip_answer = f"**분류 안내:** 본 품목은 `{rep.primary_category}` 대상입니다."
    if col_d.button("⚠️ 거부 원인?", use_container_width=True):
        st.session_state.active_chip_answer = f"**수거 위험 요소:** {rep.warning_notes or '특이 위험 사항 없음'}"

    if st.session_state.get("active_chip_answer"):
        st.info(st.session_state.active_chip_answer)

    custom_q = st.text_input("보고서 기반 추가 질문", placeholder="추가로 궁금한 점을 질문하세요...", label_visibility="collapsed")
    if custom_q:
        with st.spinner("답변 작성 중..."):
            ans = engine.ask_intelligence_fallback(custom_q, rep)
            st.success(f"🤖 **Engine Answer:** {ans}")

st.divider()
st.caption("EcoLens Intelligence Engine © 2026 — Enterprise Sustainability Protocol")
