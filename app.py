import os
import json
import base64
from typing import List, Optional
from PIL import Image, ImageDraw
import streamlit as st
from pydantic import BaseModel, Field
from groq import Groq

# ==============================================================================
# 1. Pydantic Models (Ambiguity Check & Standardized Guide)
# ==============================================================================
class ComponentCheck(BaseModel):
    is_ambiguous: bool = Field(description="단일 품목이 아닌 여러 부품이 결합된 제품인지 여부 (예: 햇반 -> 용기, 필름, 종이)")
    detected_brand_or_item: str = Field(description="감지된 대표 제품명 (예: 햇반)")
    components: List[str] = Field(description="세부 부품 리스트 (예: ['플라스틱 용기', '상부 비닐 필름', '외부 종이 띠지'])")

class InspectorCheck(BaseModel):
    material_type: str = Field(description="정확한 재질 (예: PP, PET, 종이 등)")
    label_removed: bool = Field(description="라벨/필름 제거 필요 여부")
    cap_separated: bool = Field(description="뚜껑 분리 여부")
    food_residue: bool = Field(description="음식물 잔여물 여부")
    contamination_risk: str = Field(description="오염/수거거부 위험도 (낮음, 보통, 높음)")

class RecyclingGuide(BaseModel):
    item_name: str = Field(description="분석 대상 최종 품목명 (예: 햇반 플라스틱 용기)")
    eco_score: int = Field(description="종합 Eco Score (0~100)")
    eco_grade: str = Field(description="등급 (A+, A, B, C, F)")
    material: str = Field(description="최종 재질 (Plastic, Paper, Vinyl, PET, Can, Glass, Other)")
    category: str = Field(description="배출 대분류 (예: 플라스틱류 / 일반쓰레기)")
    recycling_rate_pct: int = Field(description="재활용 가능률 (%)")
    inspector: InspectorCheck = Field(description="Inspector 검수 항목")
    rejection_risk_warning: Optional[str] = Field(description="수거 거부 위험 이유 (없으면 빈값)")
    steps: List[str] = Field(description="일관된 배출 수칙")
    cautions: List[str] = Field(description="주의사항")

# ==============================================================================
# 2. Core Service (Single Source of Truth)
# ==============================================================================
class RecyclingService:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)

    def check_ambiguity(self, item_text: str) -> ComponentCheck:
        prompt = f"""
        사용자가 '{item_text}'라는 품목을 입력했습니다.
        이 제품이 여러 부품(예: 용기, 필름, 종이포장 등)으로 구성되어 분석 대상이 모호한지 판별하세요.
        응답은 다음 JSON 스키마를 준수하세요:
        {json.dumps(ComponentCheck.model_json_schema(), ensure_ascii=False, indent=2)}
        """
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return ComponentCheck.model_validate_json(response.choices[0].message.content)

    def analyze_specific_component(self, target_item: str, location: str) -> RecyclingGuide:
        system_instruction = f"""
        당신은 자원순환 AI Inspector입니다.
        대상의 정확한 재질을 객관적으로 판별하세요. (예: 햇반 용기 -> PP/Plastic, 햇반 종이 -> Paper)
        반드시 다음 JSON 스키마를 준수하여 결과를 생성하세요:
        {json.dumps(RecyclingGuide.model_json_schema(), ensure_ascii=False, indent=2)}
        """
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"배출 지역: {location}\n분석 대상: {target_item}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return RecyclingGuide.model_validate_json(response.choices[0].message.content)

    def ask_ai_chat_locked(self, question: str, guide: RecyclingGuide) -> str:
        # 단일 진실 출처 (Context Locking)
        system_prompt = f"""
        당신은 AI Eco Assistant입니다. 
        [엄격 수칙]
        1. 당신은 새로운 분류나 추론을 하지 않습니다.
        2. 오직 아래 제공된 RecyclingGuide 데이터에 기재된 내용만을 바탕으로 답변하세요.
        3. 아래 Guide의 내용과 모순되는 답변은 절대로 금지합니다.
        4. Guide에 없는 내용을 물어보면 "현재 분석 결과에는 없는 내용입니다"라고 답변하세요.

        [RecyclingGuide Data]
        - 품목명: {guide.item_name}
        - 재질: {guide.material}
        - 카테고리: {guide.category}
        - Eco Score: {guide.eco_score} (등급: {guide.eco_grade})
        - 재활용률: {guide.recycling_rate_pct}%
        - 배출 수칙: {', '.join(guide.steps)}
        - 수거거부 위험: {guide.rejection_risk_warning or '없음'}
        """
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

# ==============================================================================
# 3. Streamlit UI Architecture
# ==============================================================================
st.set_page_config(page_title="EcoLens Platform", page_icon="🌱", layout="centered")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY 설정이 필요합니다.")
    st.stop()

service = RecyclingService(groq_api_key=GROQ_API_KEY)

st.title("🌱 EcoLens AI Inspector")
st.caption("AI 기반 단일 진실 출처(Single Source of Truth) 분리배출 정밀 분석 서비스")

# Step 1: Input
item_input = st.text_input("품목명 입력", placeholder="예: 햇반, 삼다수, 컵라면", key="user_item_input")
location = st.selectbox("지역 선택", ["전국 공통", "서울특별시 강남구", "경기도 수원시"])

if st.button("🔍 AI 검수 시작", type="primary", use_container_width=True):
    if item_input:
        with st.spinner("AI가 제품 구조 및 복합 부품 여부를 감지 중입니다..."):
            ambiguity_res = service.check_ambiguity(item_input)
            st.session_state.ambiguity_res = ambiguity_res
            st.session_state.analysis_done = False
            st.session_state.current_guide = None

# Step 2: Ambiguity Selection UI (햇반 같은 복합 제품 처리)
if "ambiguity_res" in st.session_state and st.session_state.ambiguity_res:
    amb = st.session_state.ambiguity_res
    
    if amb.is_ambiguous and not st.session_state.get("analysis_done"):
        st.warning(f"💡 **'{amb.detected_brand_or_item}'**은(는) 여러 부품으로 구성된 제품입니다. 분석할 세부 대상을 선택해 주세요.")
        
        selected_component = st.radio(
            "분석 대상 선택:",
            amb.components,
            key="selected_comp"
        )
        
        if st.button("선택한 부품 정밀 분석 →"):
            with st.spinner(f"'{selected_component}' 정밀 스캐닝 중..."):
                final_item_name = f"{amb.detected_brand_or_item} ({selected_component})"
                guide = service.analyze_specific_component(final_item_name, location)
                st.session_state.current_guide = guide
                st.session_state.analysis_done = True
                st.rerun()
    elif not amb.is_ambiguous and not st.session_state.get("analysis_done"):
        with st.spinner("정밀 스캐닝 중..."):
            guide = service.analyze_specific_component(item_input, location)
            st.session_state.current_guide = guide
            st.session_state.analysis_done = True
            st.rerun()

# Step 3: Result Display & Locked Chat
if st.session_state.get("current_guide"):
    g: RecyclingGuide = st.session_state.current_guide
    
    st.divider()
    st.success("✅ AI Inspection 완료")
    
    # Result Card
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(g.item_name)
        st.write(f"**추정 재질:** `{g.material}` | **분류:** `{g.category}`")
        st.write(f"**기본 재활용률:** `{g.recycling_rate_pct}%`")
    with col2:
        st.metric("Eco Score", f"{g.eco_score}점", delta=f"Grade {g.eco_grade}")

    # Risk Warning
    if g.rejection_risk_warning or g.recycling_rate_pct == 0:
        st.error(f"⚠️ **수거 거부 위험:** {g.rejection_risk_warning or '복합재질 또는 오염 가능성으로 수거 거부될 위험이 있습니다.'}")

    # Steps
    st.write("")
    st.markdown("##### 📋 올바른 배출 수칙")
    for i, step in enumerate(g.steps, 1):
        st.markdown(f"**{i}.** {step}")

    # Step 4: Context-Locked AI Chat
    st.divider()
    st.markdown("### 💬 AI Q&A (분석 결과 기반 100% 동기화)")
    st.caption("이 챗봇은 위 분석 결과(Guide)만을 기준으로 답변하여 모순이 발생하지 않습니다.")
    
    chat_q = st.text_input("질문하기", placeholder=f"예: '{g.item_name}'의 배출 방법 다시 설명해줘", key="locked_chat_input")
    if chat_q:
        with st.spinner("분석 결과 데이터 검증 후 답변 중..."):
            answer = service.ask_ai_chat_locked(chat_q, g)
            st.info(f"🤖 **AI 답변:** {answer}")
