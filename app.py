import streamlit as st
import numpy as np
import pandas as pd
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch

import re

# Custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📄 Scientific Paper Classifier")
st.markdown("Enter the paper's abstract and optionally the title to predict its category.")

CATEGORIES = ['cs.LG', 'hep-ph', 'hep-th', 'quant-ph', 'cs.CV', 'cs.AI', 'gr-qc', 'astro-ph', 'cond-mat.mtrl-sci', 'cond-mat.mes-hall', 'math.MP', 'math-ph', 'cs.CL', 'cond-mat.str-el', 'cond-mat.stat-mech', 'astro-ph.CO', 'math.CO', 'stat.ML', 'astro-ph.GA', 'math.AP', 'astro-ph.SR', 'astro-ph.HE', 'math.PR', 'nucl-th', 'hep-ex', 'math.AG', 'math.OC', 'physics.optics', 'cs.IT', 'math.IT', 'cond-mat.supr-con', 'math.NT', 'math.DG', 'cond-mat.soft', 'math.NA', 'cs.RO', 'math.DS', 'cs.CR', 'cs.SY', 'math.FA', 'eess.SP', 'astro-ph.IM', 'astro-ph.EP', 'physics.flu-dyn', 'stat.ME', 'hep-lat', 'eess.SY', 'cs.NA', 'math.RT', 'eess.IV', 'nucl-ex', 'cs.DS', 'physics.comp-ph', 'stat.TH', 'math.ST', 'cond-mat.dis-nn', 'cs.NI', 'cs.DC', 'physics.chem-ph', 'math.GT', 'math.GR', 'math.CA', 'physics.soc-ph', 'cs.HC', 'cs.CY', 'physics.ins-det', 'cond-mat.quant-gas', 'physics.atom-ph', 'cs.SI', 'physics.app-ph', 'stat.AP', 'cs.IR', 'cs.SE', 'math.QA', 'math.RA', 'math.CV', 'physics.plasm-ph', 'eess.AS', 'cs.LO', 'cs.SD', 'math.AT', 'physics.bio-ph', 'nlin.CD', 'cond-mat.other', 'cs.NE', 'cs.DM', 'cond-mat', 'math.LO', 'math.AC', 'math.OA', 'cs.GT', 'nlin.SI', 'q-bio.PE', 'math.MG', 'cs.CC', 'q-bio.QM', 'physics.data-an', 'physics.gen-ph', 'q-bio.NC', 'math.SP', 'nlin.PS', 'cs.DB', 'math.SG', 'math.CT', 'cs.MA', 'stat.CO', 'physics.class-ph', 'cs.PL', 'cs.CE', 'physics.acc-ph', 'physics.med-ph', 'physics.geo-ph', 'cs.MM', 'nlin.AO', 'cs.GR', 'cs.CG', 'physics.ao-ph', 'physics.space-ph', 'math.KT', 'cs.AR', 'q-bio.BM', 'q-fin.EC', 'cs.ET', 'math.GN', 'cs.FL', 'cs.DL', 'physics.hist-ph', 'econ.GN', 'cs.PF', 'econ.EM', 'math.GM', 'physics.ed-ph', 'q-bio.MN', 'math.HO', 'q-fin.ST', 'q-bio.GN', 'econ.TH', 'physics.atm-clus', 'q-fin.MF', 'q-fin.GN', 'cs.SC', 'q-fin.CP', 'physics.pop-ph', 'q-fin.RM', 'q-bio.TO', 'chao-dyn', 'cs.MS', 'q-bio.CB', 'cs.OH', 'q-fin.PR', 'q-fin.PM']
MODEL_PATH = 'model'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(CATEGORIES)).to(DEVICE, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    return model, tokenizer

def top_p(probs, p=0.95):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_keep = cumulative_probs <= p
    return sorted_indices[sorted_indices_to_keep]


def top_k(probs, k=10):
    topk = torch.topk(probs[0], k=k, dim=-1)
    return topk.indices

with st.form('form'):
    title = st.text_input("Paper Title (required):")
    abstract = st.text_area("Abstract (optional):", height=150)
    # Create placeholder containers
    results_placeholder = st.empty()
    chart_placeholder = st.empty()
    error_placeholder = st.empty()

    button = st.form_submit_button("Classify Paper")

model, tokenizer = load_model()

def predict_category(abstract, title):
    abstract = re.sub(r"[^\w\d'\s]+", " ", abstract.lower().strip())
    title = re.sub(r"[^\w\d'\s]+", " ", title.lower().strip())
    to_tokenize = title + "[SEP]" + abstract
    tokenized = tokenizer(to_tokenize, return_tensors="pt", padding=True, max_length=512, truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**tokenized)
        probabilities = outputs.logits.softmax(dim=-1)

        top_p_index = top_k(probabilities, k=10)
        cats = list(map(CATEGORIES.__getitem__, top_p_index))

        return dict(zip(cats, probabilities[0, top_p_index].to(dtype=torch.float16).detach().cpu().numpy()))

if button:
    error_placeholder.empty()  # Clear previous errors
    results_placeholder.empty()  # Clear previous results
    chart_placeholder.empty()  # Clear previous chart
    
    if not title.strip():
        error_placeholder.error("Please enter an title to classify.")
    else:        
        with st.spinner("Analyzing paper..."):
            probabilities = predict_category(abstract.strip(), title.strip())
            sorted_probs = sorted(probabilities.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)
            
            # Display results in the placeholder
            with results_placeholder.container():
                st.subheader("📊 Prediction Results")
                
                top_cats = []
                for i, (cat, prob) in enumerate(sorted_probs[:3]):
                    top_cats.append({
                        "Rank": f"{i+1}.",
                        "Category": cat,
                        "Probability": f"{prob:.1%}"
                    })
                    
                st.table(pd.DataFrame(top_cats).set_index("Rank"))
            
            # Display chart in its placeholder
            with chart_placeholder.container():
                st.markdown("### Probability Distribution")
                prob_df = pd.DataFrame({
                    "Category": probabilities.keys(),
                    "Probability": probabilities.values()
                })
                
                st.bar_chart(prob_df.set_index("Category"))