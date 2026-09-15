#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from predict_all import run_all
from reup_ai import CLASS_NAMES, COMPONENT_CLASS_NAMES


DISPLAY = {
    "unable_to_determine": "Unable to determine",
    "no_visible_damage": "No visible damage detected",
    "visible_damage_detected": "Visible damage detected",
}


def label(value: str) -> str:
    return DISPLAY.get(value, value.replace("_", " ").title())


def default_index(options: list[str], suggestion: str) -> int:
    return options.index(suggestion) if suggestion in options else 0


st.set_page_config(page_title="ReUP AI Listing Assistant", page_icon="♻️", layout="wide")
st.title("ReUP AI Listing Assistant")
st.caption("AI suggestions for human verification — not a structural or safety assessment")

uploaded = st.file_uploader("Upload one clear construction-component image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to generate material, component, and visible-condition suggestions.")
    st.stop()

suffix = Path(uploaded.name).suffix or ".jpg"
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
    temporary.write(uploaded.getbuffer())
    image_path = Path(temporary.name)

with st.spinner("Analyzing the image…"):
    result = run_all(image_path)

left, right = st.columns([1, 1.25])
with left:
    st.image(uploaded, caption="Uploaded listing image", use_container_width=True)

with right:
    material = result["fields"]["material_type"]
    component = result["fields"]["component_type"]
    condition = result["fields"]["visible_condition"]
    material_options = ["unable_to_determine", *CLASS_NAMES]
    component_options = ["unable_to_determine", *COMPONENT_CLASS_NAMES]
    condition_options = ["unable_to_determine", "no_visible_damage", "visible_damage_detected"]

    with st.form("verification"):
        st.subheader("Verify the suggested listing fields")
        selected_material = st.selectbox(
            "Material type", material_options,
            index=default_index(material_options, material["suggestion"]), format_func=label,
        )
        selected_component = st.selectbox(
            "Component type", component_options,
            index=default_index(component_options, component["suggestion"]), format_func=label,
        )
        selected_condition = st.selectbox(
            "Visible condition", condition_options,
            index=default_index(condition_options, condition["suggestion"]), format_func=label,
            help="The model checks for visible damage cues. It does not determine structural safety or hidden damage.",
        )
        verified = st.checkbox("I reviewed and verified these three fields")
        submitted = st.form_submit_button("Use verified information", type="primary")

    for title, field in (("Material", material), ("Component", component), ("Condition", condition)):
        if field["status"] == "model_not_trained":
            st.warning(f"{title} model is not trained yet; please select the field manually.")
        elif field["status"] == "low_confidence":
            st.warning(f"{title} prediction was below its confidence threshold; please select manually.")

    if submitted:
        if not verified:
            st.error("Please review the suggestions and check the verification box.")
        else:
            verified_record = {
                "material_type": selected_material,
                "component_type": selected_component,
                "visible_condition": selected_condition,
                "verified_by_human": True,
                "ai_suggestions": result["fields"],
            }
            st.success("Verified information is ready for the ReUP listing form.")
            st.download_button(
                "Download verified JSON",
                data=json.dumps(verified_record, indent=2),
                file_name="reup_verified_listing_fields.json",
                mime="application/json",
            )
