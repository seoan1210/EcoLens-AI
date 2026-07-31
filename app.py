import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel, Field
from groq import Groq

# ==============================================================================
# 1. 상세 분석 보고서용 Pydantic 스키마 정의
# ==============================================================================
class SubComponent(BaseModel):
    name: str = Field(description="부품명 (예: 플라스틱 용기, 비닐 필름, 종이 띠지)")
    material: str = Field(description="재질 (PP, PET, Paper, Vinyl, Other)")
    recycling_rate: int = Field(description="재활용률 (%)")
    status: str = Field(description="검사 상태 (PASS, WARNING, FAIL)")

class InspectionChecklist(BaseModel):
    object_detection: str = Field(description="객체 인식 상태 (PASS, WARNING, FAIL)")
    material_analysis: str = Field(description="재질 정밀 감지 상태 (PASS, WARNING, FAIL)")
    residue_detection: str = Field(description="잔여물 검수 상태 (PASS, WARNING, FAIL)")
    label_detection: str = Field(description="라벨/필름 분리 검수 상태 (PASS, WARNING, FAIL)")
    cap_detection: str = Field(description="뚜껑/이물질 검수 상태 (PASS, WARNING, FAIL)")
    local_policy: str = Field(description="지자체 정책 준수 상태 (PASS, WARNING, FAIL)")

class ScoreBreakdown(BaseModel):
    material_score: int = Field(description="재질 평가 점수 (max 25)")
    cleaning_score: int = Field(description="세척 상태 점수 (max 20)")
    contamination_score: int = Field(description="오염도 점수 (max 30)")
    policy_score: int = Field(description="규정 적합 점수 (max 25)")

class InspectionReport(BaseModel):
    detected_item: str = Field(description="감지된 품목 대표명")
    is_ambiguous: bool = Field(description="복합 제품 여부")
    components: List[SubComponent] = Field(description="구성 부품 리스트")
    default_component_index: int = Field(description="기본 선택 부품 인덱스")
    
    total_score: int = Field(description="총점 (0~100)")
    grade: str = Field(description="AAA, AA, A, B, C 중 선택")
    score_breakdown: ScoreBreakdown = Field(description="세부 점수")
    checklist: InspectionChecklist = Field(description="6대 검사 스탬프")
    
    primary_category: str = Field(description="배출 카테고리 (예: 플라스틱류 / 일반쓰레기)")
    detailed_summary: str = Field(description="종합 진단 총평 (3~4문장 이상의 매우 상세한 설명)")
    score_reasoning: str = Field(description="점수 감점 및 부여에 대한 상세한 이유 설명")
    steps: List[str] = Field(description="단계별 실천 수칙 (최소 4단계 이상 아주 상세히 작성)")
    warning_notes: Optional[str] = Field(description="주의 사항 및 환경적 영향 설명")

# ==============================================================================
# 2. EcoLens Intelligence Engine
# ==============================================================================
class EcoLensEngine:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)

    def run_single_pass_inspection(self, item_text: str, location: str) -> InspectionReport:
        system_prompt = f"""
        당신은 자원순환 AI 검수 엔진 'EcoLens Intelligence Engine'입니다.
        입력된 제품에 대해 전문 분석 보고서를 작성하세요.
        
        [필수 지침]
        1. 단순한 요약이 아니라 매우 '상세하고 정밀한 분석 내용'을 작성해야 합니다.
        2. detailed_summary에는 재질 특성, 분리배출 필요성, 환경적 유의점까지 상세히 기술하세요.
        3. score_reasoning에는 왜 이 점수가 나왔는지(세척 미흡, 라벨 혼합 등) 감점 이유를 구체적으로 적으세요.
        4. steps는 누구나 쉽게 따라 할 수 있도록 구체적인 세척, 분리, 건조, 배출 방법을 4~5단계로 나누어 설명하세요.
        
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
# 3. UI Setup
# ==============================================================================
st.set_page_config(page_title="EcoLens Infographic Generator", page_icon="🌱", layout="centered")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY 설정이 필요합니다.")
    st.stop()

engine = EcoLensEngine(groq_api_key=GROQ_API_KEY)

st.title("🌱 EcoLens Infographic Generator")

target_input = st.text_input("검사 대상 입력", placeholder="예: 햇반, 도시락, 컵라면, 삼다수")
location = st.selectbox("배출 규정 지역", ["전국 공통 기준", "서울특별시 강남구", "경기도 수원시"])

if st.button("✨ 캔바 스타일 상세 보고서 생성하기", type="primary", use_container_width=True):
    if target_input:
        with st.spinner("상세 보고서를 분석 및 디자인하는 중..."):
            try:
                report = engine.run_single_pass_inspection(target_input, location)
                st.session_state.report = report
                st.session_state.selected_comp_idx = report.default_component_index
                st.session_state.doc_id = f"CANVA-{uuid.uuid4().hex[:6].upper()}"
                st.session_state.created_at = datetime.now().strftime("%Y.%m.%d")
            except Exception as e:
                st.error(f"생성 실패: {e}")

# ==============================================================================
# 4. 보고서 렌더링 (HTML 깨짐 보장)
# ==============================================================================
if "report" in st.session_state and st.session_state.report:
    rep: InspectionReport = st.session_state.report
    doc_id = st.session_state.get("doc_id", "CANVA-001")
    created_at = st.session_state.get("created_at", "2026.04.04")

    if rep.is_ambiguous and rep.components:
        st.write("")
        st.caption("💡 세부 구성품 선택:")
        comp_names = [f"{c.name} ({c.material})" for c in rep.components]
        selected_tab = st.radio("구성 요소 선택", comp_names, horizontal=True, label_visibility="collapsed")
        st.session_state.selected_comp_idx = comp_names.index(selected_tab)

    active_comp = rep.components[st.session_state.selected_comp_idx] if rep.components else None
    sb = rep.score_breakdown
    chk = rep.checklist

    def get_stamp_html(status: str):
        if status == "PASS": return '<span style="background:#DCFCE7; color:#15803D; padding:4px 8px; border-radius:6px; font-weight:bold; font-size:12px;">PASS</span>'
        elif status == "WARNING": return '<span style="background:#FEF9C3; color:#A16207; padding:4px 8px; border-radius:6px; font-weight:bold; font-size:12px;">WARNING</span>'
        return '<span style="background:#FEE2E2; color:#B91C1C; padding:4px 8px; border-radius:6px; font-weight:bold; font-size:12px;">FAIL</span>'

    steps_list_html = "".join([f"<li style='margin-bottom:6px;'>{s}</li>" for s in rep.steps])

    # Canva 카드를 깔끔한 HTML 코드 세트로 조립
    html_card = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
            body {{ font-family: 'Pretendard', sans-serif; margin:0; padding:10px; background:#F1F5F9; color:#0F172A; }}
            .card {{ background:#FFF; border-radius:24px; overflow:hidden; box-shadow:0 15px 30px rgba(0,0,0,0.08); border:1px solid #E2E8F0; }}
            .header {{ background:linear-gradient(135deg, #059669 0%, #10B981 100%); padding:28px; color:#FFF; display:flex; justify-content:space-between; align-items:center; }}
            .title {{ font-size:22px; font-weight:900; margin-top:4px; }}
            .sub-title {{ font-size:12px; font-weight:700; opacity:0.85; letter-spacing:1px; }}
            .badge {{ background:rgba(255,255,255,0.2); border:1px solid rgba(255,255,255,0.4); padding:10px 18px; border-radius:16px; text-align:center; }}
            .score {{ font-size:32px; font-weight:900; line-height:1; }}
            .body {{ padding:24px; }}
            .section {{ background:#F8FAFC; border-radius:16px; padding:18px; margin-bottom:16px; border:1px solid #E2E8F0; }}
            .sec-title {{ font-size:13px; font-weight:800; color:#64748B; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.5px; }}
            .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; }}
            .row {{ display:flex; justify-content:space-between; align-items:center; font-size:13px; padding:6px 0; border-bottom:1px dashed #E2E8F0; }}
            .bar-bg {{ background:#E2E8F0; height:6px; border-radius:3px; margin-top:4px; overflow:hidden; }}
            .bar-fill {{ background:#10B981; height:100%; border-radius:3px; }}
            .desc-text {{ font-size:13.5px; line-height:1.6; color:#334155; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <div>
                    <div class="sub-title">ECOLENS REPORT • {doc_id}</div>
                    <div class="title">{rep.detected_item}</div>
                    <div style="font-size:13px; margin-top:6px; opacity:0.95;">
                        분류: <b>{rep.primary_category}</b> {f'({active_comp.name} / {active_comp.material})' if active_comp else ''}
                    </div>
                </div>
                <div class="badge">
                    <div class="score">{rep.total_score}</div>
                    <div style="font-size:11px; font-weight:800; margin-top:2px;">GRADE {rep.grade}</div>
                </div>
            </div>

            <div class="body">
                <!-- 1. 종합 진단 및 감지 결과 상세 -->
                <div class="section">
                    <div class="sec-title">📌 종합 분석 리포트</div>
                    <div class="desc-text">{rep.detailed_summary}</div>
                </div>

                <!-- 2. 점수 세부 분석 및 검사 항목 -->
                <div class="grid">
                    <div class="section" style="margin-bottom:0;">
                        <div class="sec-title">📊 점수 상세</div>
                        <div style="margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:bold;"><span>재질 평가</span><span>{sb.material_score}/25</span></div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{(sb.material_score/25)*100}%;"></div></div>
                        </div>
                        <div style="margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:bold;"><span>세척 상태</span><span>{sb.cleaning_score}/20</span></div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{(sb.cleaning_score/20)*100}%;"></div></div>
                        </div>
                        <div style="margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:bold;"><span>오염도 점수</span><span>{sb.contamination_score}/30</span></div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{(sb.contamination_score/30)*100}%;"></div></div>
                        </div>
                        <div>
                            <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:bold;"><span>규정 적합성</span><span>{sb.policy_score}/25</span></div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{(sb.policy_score/25)*100}%;"></div></div>
                        </div>
                    </div>

                    <div class="section" style="margin-bottom:0;">
                        <div class="sec-title">🔍 6대 검사 스탬프</div>
                        <div class="row"><span>객체 인식</span>{get_stamp_html(chk.object_detection)}</div>
                        <div class="row"><span>재질 분석</span>{get_stamp_html(chk.material_analysis)}</div>
                        <div class="row"><span>잔여물 검수</span>{get_stamp_html(chk.residue_detection)}</div>
                        <div class="row"><span>라벨 분리</span>{get_stamp_html(chk.label_detection)}</div>
                        <div class="row"><span>뚜껑/이물질</span>{get_stamp_html(chk.cap_detection)}</div>
                        <div class="row"><span>지자체 규정</span>{get_stamp_html(chk.local_policy)}</div>
                    </div>
                </div>

                <!-- 3. 감지 분석 상세 이유 -->
                <div class="section" style="margin-top:16px;">
                    <div class="sec-title">💡 점수 산정 & 평가 이유</div>
                    <div class="desc-text">{rep.score_reasoning}</div>
                </div>

                <!-- 4. 상세 배출 지침 -->
                <div class="section" style="background:#EFF6FF; border-color:#BFDBFE; margin-bottom:0;">
                    <div class="sec-title" style="color:#1E40AF;">📋 단계별 올바른 분리배출 가이드</div>
                    <ol class="desc-text" style="padding-left:18px; margin:0;">
                        {steps_list_html}
                    </ol>
                    {f'<div style="margin-top:12px; padding:10px; background:#FEE2E2; border-radius:8px; color:#991B1B; font-size:12px; font-weight:bold;">⚠️ 주의: {rep.warning_notes}</div>' if rep.warning_notes else ''}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # Streamlit에 HTML 카드 렌더링
    components.html(html_card, height=850, scrolling=True)
