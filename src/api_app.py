from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, model_validator

from reup_ai.inference_service import ImageInferenceService


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "material_type": ROOT / "model" / "best_model.pt",
    "component_type": ROOT / "model" / "component" / "best_model.pt",
    "visible_condition": ROOT / "model" / "condition" / "best_model.pt",
}
THRESHOLDS = {
    "material_type": 0.65,
    "component_type": 0.60,
    "visible_condition": 0.65,
}

with (ROOT / "data" / "a1_a3_factors.json").open(encoding="utf-8") as stream:
    FACTOR_CATALOG = json.load(stream)
FACTORS = {factor["factor_id"]: factor for factor in FACTOR_CATALOG["factors"]}

with (ROOT / "results" / "ai_layer_metrics.json").open(encoding="utf-8") as stream:
    AI_METRICS = json.load(stream)

INFERENCE = ImageInferenceService(MODEL_PATHS, THRESHOLDS)
ANALYSES: dict[str, dict] = {}


class VerifiedFields(BaseModel):
    material_type: str = Field(min_length=1, max_length=80)
    component_type: str = Field(min_length=1, max_length=80)
    visible_condition: str = Field(min_length=1, max_length=120)
    verified_by_human: bool


class AssessmentRequest(BaseModel):
    analysis_id: str | None = None
    verified_fields: VerifiedFields
    quantity: float = Field(gt=0, le=1_000_000)
    unit: str = Field(pattern="^(m3|kg)$")
    factor_id: str

    @model_validator(mode="after")
    def require_human_verification(self):
        if not self.verified_fields.verified_by_human:
            raise ValueError("Human verification is required before calculation.")
        return self


app = FastAPI(
    title="ReUP Capstone AI and A1-A3 Assessment API",
    version="1.0.0-demo",
    description=(
        "Standalone capstone backend for human-verified image information extraction "
        "and transparent A1-A3 product-stage assessment."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/demo")


@app.get("/health")
def health():
    return {
        "status": "ready",
        "service": "reup-capstone-backend",
        "version": app.version,
        "models": INFERENCE.model_status(),
        "factor_catalog_version": FACTOR_CATALOG["catalog_version"],
        "factor_count": len(FACTORS),
        "assessment_boundary": FACTOR_CATALOG["assessment_boundary"],
    }


@app.get("/api/v1/models/metrics")
def model_metrics():
    return AI_METRICS


@app.get("/api/v1/factors")
def list_factors(material_type: str | None = None, unit: str | None = None):
    factors = list(FACTORS.values())
    if material_type:
        factors = [f for f in factors if f["material_type"] == material_type]
    if unit:
        factors = [f for f in factors if f["denominator_unit"] == unit]
    return {
        "catalog_version": FACTOR_CATALOG["catalog_version"],
        "assessment_boundary": FACTOR_CATALOG["assessment_boundary"],
        "factors": factors,
    }


@app.post("/api/v1/analyze-image")
async def analyze_image(
    image: Annotated[UploadFile, File(description="JPEG or PNG listing image")]
):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    content = await image.read()
    try:
        result = INFERENCE.analyze(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    analysis_id = str(uuid.uuid4())
    response = {
        "analysis_id": analysis_id,
        "filename": image.filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "Suggestions require human verification before listing use.",
        **result,
    }
    ANALYSES[analysis_id] = response
    if len(ANALYSES) > 100:
        ANALYSES.pop(next(iter(ANALYSES)))
    return response


@app.post("/api/v1/assessments/a1-a3")
def calculate_a1_a3(request: AssessmentRequest):
    factor = FACTORS.get(request.factor_id)
    if factor is None:
        raise HTTPException(status_code=404, detail="The selected factor was not found.")
    if factor["status"] != "approved_for_demo":
        raise HTTPException(status_code=422, detail="The selected factor is not approved.")
    if request.unit != factor["denominator_unit"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unit mismatch: factor requires {factor['denominator_unit']}, "
                f"but the request supplied {request.unit}."
            ),
        )
    if request.verified_fields.material_type != factor["material_type"]:
        raise HTTPException(
            status_code=422,
            detail="Verified material type does not match the selected factor.",
        )
    if request.analysis_id and request.analysis_id not in ANALYSES:
        raise HTTPException(status_code=404, detail="The referenced analysis was not found.")

    value = request.quantity * factor["factor_value_kgco2e"]
    assessment_id = str(uuid.uuid4())
    return {
        "assessment_id": assessment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": request.analysis_id,
        "verified_fields": request.verified_fields.model_dump(),
        "assessment_boundary": "A1-A3 product stage only",
        "quantity": request.quantity,
        "unit": request.unit,
        "factor": factor,
        "formula": "verified_quantity * a1_a3_factor",
        "calculation": (
            f"{request.quantity:g} {request.unit} * "
            f"{factor['factor_value_kgco2e']:g} kg CO2e/{request.unit}"
        ),
        "estimated_potential_avoided_a1_a3_kgco2e": round(value, 4),
        "result_status": "screening_estimate",
        "claim": (
            "Estimated potential avoided new-material product-stage impact; "
            "not confirmed carbon saved."
        ),
        "verification_required": False,
    }


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo_page():
    return HTMLResponse(DEMO_HTML)


DEMO_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ReUP Capstone AI Demo</title>
  <style>
    :root { --navy:#0b2942; --blue:#1674a9; --green:#198754; --pale:#eef6fa; --ink:#17212b; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,Arial,sans-serif; color:var(--ink); background:#f4f7f8; }
    header { background:linear-gradient(135deg,var(--navy),#135d7e); color:white; padding:28px max(24px,calc((100vw - 1120px)/2)); }
    header h1 { margin:0 0 8px; font-size:clamp(24px,4vw,38px); }
    header p { margin:0; opacity:.9; }
    main { max-width:1120px; margin:24px auto; padding:0 20px 40px; }
    .grid { display:grid; grid-template-columns:1fr 1.25fr; gap:22px; align-items:start; }
    .card { background:white; border-radius:14px; padding:22px; box-shadow:0 6px 24px rgba(14,42,63,.08); margin-bottom:20px; }
    h2 { margin:0 0 16px; color:var(--navy); font-size:21px; }
    label { display:block; font-weight:700; margin:14px 0 6px; }
    input,select,button { width:100%; padding:11px 12px; border:1px solid #bccbd3; border-radius:8px; font-size:15px; }
    button { margin-top:16px; background:var(--blue); color:white; border:0; font-weight:800; cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    img { display:none; width:100%; max-height:420px; object-fit:contain; border-radius:10px; background:#edf2f4; }
    .field { padding:11px 12px; border-left:4px solid var(--blue); background:var(--pale); margin:9px 0; border-radius:6px; }
    .confidence { color:#4f6674; font-size:13px; margin-top:4px; }
    .notice { background:#fff6df; border:1px solid #eed28a; padding:12px; border-radius:8px; font-size:14px; }
    .success { background:#eaf7ef; border-color:#9fd5b2; }
    .result { font-size:34px; font-weight:900; color:var(--green); margin:10px 0; }
    .small { font-size:13px; color:#58707d; }
    .hidden { display:none; }
    @media (max-width:760px) { .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header>
  <h1>ReUP Explainable AI Listing Assistant</h1>
  <p>Image suggestions + human verification + transparent A1-A3 assessment</p>
</header>
<main>
  <div class="grid">
    <section class="card">
      <h2>1. Upload a material image</h2>
      <input id="image" type="file" accept="image/jpeg,image/png">
      <button id="analyze">Analyze image</button>
      <p id="status" class="small"></p>
      <img id="preview" alt="Uploaded construction material">
    </section>
    <section class="card">
      <h2>2. Review AI suggestions</h2>
      <div class="notice">AI suggestions are preliminary. A person must verify all three fields; visible condition is not a structural or safety assessment.</div>
      <div id="suggestions"><p class="small">Upload an image to begin.</p></div>
      <div id="verification" class="hidden">
        <label for="material">Verified material type</label><select id="material"></select>
        <label for="component">Verified component type</label><select id="component"></select>
        <label for="condition">Verified visible condition</label><select id="condition"></select>
        <label><input id="verified" type="checkbox" style="width:auto"> I reviewed and verified these fields</label>
      </div>
    </section>
  </div>
  <section id="assessment" class="card hidden">
    <h2>3. Calculate estimated potential A1-A3 impact</h2>
    <div class="grid">
      <div>
        <label for="factor">Governed demo factor</label><select id="factor"></select>
        <label for="quantity">Verified quantity</label><input id="quantity" type="number" min="0.001" step="any" value="5">
        <button id="calculate">Calculate A1-A3 estimate</button>
      </div>
      <div id="calculation"><p class="small">Select a compatible factor and verified quantity.</p></div>
    </div>
  </section>
</main>
<script>
let analysisId = null, fields = null, factors = [];
const labels = x => x.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
const optionList = (select, values, selected) => {
  select.innerHTML = values.map(v=>`<option value="${v}" ${v===selected?'selected':''}>${labels(v)}</option>`).join('');
};
async function loadFactors() {
  const r = await fetch('/api/v1/factors'); factors = (await r.json()).factors;
}
loadFactors();
document.getElementById('image').addEventListener('change', e => {
  const file=e.target.files[0], img=document.getElementById('preview');
  if(file){ img.src=URL.createObjectURL(file); img.style.display='block'; }
});
document.getElementById('analyze').onclick = async () => {
  const file=document.getElementById('image').files[0]; if(!file){alert('Choose a JPEG or PNG image.');return;}
  const btn=document.getElementById('analyze'), status=document.getElementById('status');
  btn.disabled=true; status.textContent='Running three trained models...';
  const form=new FormData(); form.append('image',file);
  try {
    const r=await fetch('/api/v1/analyze-image',{method:'POST',body:form}); const data=await r.json();
    if(!r.ok) throw new Error(data.detail||'Analysis failed');
    analysisId=data.analysis_id; fields=data.fields;
    document.getElementById('suggestions').innerHTML=Object.entries(fields).map(([k,v])=>
      `<div class="field"><strong>${labels(k)}:</strong> ${labels(v.suggestion)}<div class="confidence">Confidence ${(v.confidence*100).toFixed(1)}% · threshold ${(v.confidence_threshold*100).toFixed(0)}% · human verification required</div></div>`).join('');
    optionList(document.getElementById('material'),Object.keys(fields.material_type.probabilities),fields.material_type.raw_prediction);
    optionList(document.getElementById('component'),Object.keys(fields.component_type.probabilities),fields.component_type.raw_prediction);
    optionList(document.getElementById('condition'),Object.keys(fields.visible_condition.probabilities),fields.visible_condition.raw_prediction);
    document.getElementById('verification').classList.remove('hidden'); document.getElementById('assessment').classList.remove('hidden');
    const compatible=factors.filter(f=>f.material_type===document.getElementById('material').value);
    document.getElementById('factor').innerHTML=compatible.length?compatible.map(f=>`<option value="${f.factor_id}">${labels(f.material_type)} · ${f.factor_value_kgco2e} kg CO2e/${f.denominator_unit}</option>`).join(''):'<option value="">No governed demo factor for this material</option>';
    status.textContent='Analysis complete. Review every field.';
  } catch(e){ status.textContent=e.message; } finally { btn.disabled=false; }
};
document.getElementById('material').onchange=()=>{
  const compatible=factors.filter(f=>f.material_type===document.getElementById('material').value);
  document.getElementById('factor').innerHTML=compatible.length?compatible.map(f=>`<option value="${f.factor_id}">${labels(f.material_type)} · ${f.factor_value_kgco2e} kg CO2e/${f.denominator_unit}</option>`).join(''):'<option value="">No governed demo factor for this material</option>';
};
document.getElementById('calculate').onclick=async()=>{
  if(!document.getElementById('verified').checked){alert('Verify the three fields first.');return;}
  const factor=factors.find(f=>f.factor_id===document.getElementById('factor').value); if(!factor){alert('No compatible governed demo factor is available.');return;}
  const payload={analysis_id:analysisId,verified_fields:{material_type:document.getElementById('material').value,component_type:document.getElementById('component').value,visible_condition:document.getElementById('condition').value,verified_by_human:true},quantity:Number(document.getElementById('quantity').value),unit:factor.denominator_unit,factor_id:factor.factor_id};
  const r=await fetch('/api/v1/assessments/a1-a3',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const data=await r.json();
  if(!r.ok){alert(data.detail||'Calculation failed');return;}
  document.getElementById('calculation').innerHTML=`<div class="notice success"><strong>A1-A3 product-stage estimate</strong><div class="result">${data.estimated_potential_avoided_a1_a3_kgco2e.toLocaleString()} kg CO2e</div><div>${data.calculation}</div><p class="small">${data.claim}</p></div>`;
};
</script>
</body>
</html>
"""
