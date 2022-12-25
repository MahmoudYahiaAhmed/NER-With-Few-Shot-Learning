from glob import glob
import streamlit as st
import spacy
from spacy import displacy
from transformers import pipeline
import re
from spacy.matcher import DependencyMatcher 
from spacy import displacy
from spacy.pipeline import EntityRuler
import pickle
import pandas as pd


nlp = spacy.load("en_core_web_sm", disable=["ner"])
ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

zero_shot_classifier = pipeline("zero-shot-classification",
                                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

colors = ["#7aecec", "#bfeeb7", "#feca74", "#ff9561", "#aa9cfc", "#c887fb", "#9cc9cc", "#ffeb80",
          "#ff8197", "#ff8197", "#f0d0ff", "#bfe1d9", "#bfe1d9", "#e4e7d2", "#e4e7d2", "#e4e7d2",
          "#e4e7d2", "#e4e7d2"]

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

def clean(txt):
    '''
    Elminate any non word character from the input text, and remove any additionl spaces.
    
    Args:
      - txt (string) -> the unclean text.
    '''
   #     txt = re.sub(r'\W', ' ', txt)
   #     txt = re.sub(r' \w ', ' ', txt)
   #     txt = re.sub(r' +', ' ', txt)
    return txt.strip().lower()


patterns = [
    # ADJ -> NOUN
    [{"RIGHT_ID": "adj", "RIGHT_ATTRS": {"POS": "ADJ"}},
     {"LEFT_ID": "adj", "REL_OP": "<", "RIGHT_ID": "subject", "RIGHT_ATTRS": { "POS": "NOUN"}}],
            
    # NOUN
    [{"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}}],
    
    # NOUN . NOUN
    [{"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}},
     {"LEFT_ID": "noun", "REL_OP": ".", "RIGHT_ID": "subject", "RIGHT_ATTRS": { "POS": "NOUN"}}],
    
    # PROPN
    [{"RIGHT_ID": "pnoun","RIGHT_ATTRS": {"POS": "PROPN"}}],
    
    # PROPN << NOUN
    [{"RIGHT_ID": "pnoun","RIGHT_ATTRS": {"POS": "PROPN"}},
     {"LEFT_ID": "pnoun", "REL_OP": "<<", "RIGHT_ID": "subject", "RIGHT_ATTRS": { "POS": "NOUN"}}],
    
    # PROPN << PROPN
    [{"RIGHT_ID": "pnoun", "RIGHT_ATTRS": {"POS": "PROPN"}},
     {"LEFT_ID": "pnoun", "REL_OP": "<<", "RIGHT_ID": "subject", "RIGHT_ATTRS": { "POS": "PROPN"}}],
    
    # VERB < NOUN
    [{"RIGHT_ID": "verb","RIGHT_ATTRS": {"POS": "VERB"}},
     {"LEFT_ID": "verb", "REL_OP": "<", "RIGHT_ID": "subject", "RIGHT_ATTRS": { "POS": "NOUN"}}],
]


header = st.container()
sreachRecomend = st.container()
productInfo = st.container()
simalerProduct = st.container()


if 'count' not in st.session_state:
	st.session_state['extracted_ingrediants'] = None


# Load the saved model
with open("pre_process_pip.pkl", 'rb') as file:
    loaded_tfidf_pip = pickle.load(file)

# Load the saved model
with open("classification_svg_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

# Load the saved model
with open("lb_encoder.pkl", 'rb') as file:
    loaded_lb_encoder = pickle.load(file)

# Load the saved model
with open("km_model.pkl", 'rb') as file:
    loaded_km_model = pickle.load(file)

# Load the saved model
with open("km_tfidf_pip.pkl", 'rb') as file:
    loaded_km_tfidf_pip = pickle.load(file)

loaded_dishes = pd.read_csv('dishes_with_clusters.csv')
loaded_indDF = pd.read_csv('./data/indian_food.csv.xls')


btnClick=False
with header:
   with st.form('my_form'):
      st.title("Welcome to  dine and ......")
      dataQuery = st.text_area('tell me what you want to eat ?')
      entities= st.text_area('cateogies')
      conf_th = st.text_area('Confidence level')

      sumbitted = st.form_submit_button("Get Entities")
      if sumbitted:
         text = dataQuery
         org_entities = eval(entities)

         org_entities_map = {v:k for k in org_entities.keys() for v in org_entities[k]}
         for k in org_entities.keys():
            org_entities_map[k] = k

         entities = sum(org_entities.values(), []) + list(org_entities.keys())

         CONF_TH = float(conf_th)

         clean_text = clean(text)
         
         doc = nlp(clean_text)

         matcher = DependencyMatcher(nlp.vocab)
         matcher.add("entities", patterns)

         
         matches = matcher(doc)
         spans = []
         for match_id, start in matches:
            if len(start) > 1:
               start, end = start[0], start[1]+1
            else:
               start, end = start[0], start[0]+1
            spans.append(doc[start:end])
         filterd_spans = list(spacy.util.filter_spans(spans))

         results = zero_shot_classifier([str(s) for s in filterd_spans], list(entities))


         def get_highest_score_label(res):
            scores = {}
            for sc, label in zip(res['scores'], res['labels']):
               org_key = org_entities_map[label]
               scores[org_key] = scores.get(org_key, 0) + sc
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[0]

         new_results = []
         for res in results:
            label, score = get_highest_score_label(res)
            new_results.append({"sequence": res['sequence'], "labels": [label], "scores": [score]})
         results = new_results

         results_map = {i['sequence']:i for i in results}

         options = {"colors": {k:v for k,v in zip(list(org_entities.keys()), colors)},
                     "ents": list(org_entities.keys()),
                     'distance': 90}

         nlp.remove_pipe("entity_ruler")
         ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
         extracted_ingrediants = []
         for res in results:
            if res['scores'][0]> CONF_TH:
               ruler.add_patterns( [{"label": res['labels'][0], "pattern": [{"TEXT": t} for t in res['sequence'].split()]}] )
               extracted_ingrediants.append(res['sequence'])
         
         print(', '.join(extracted_ingrediants).strip())
         st.session_state.extracted_ingrediants = ', '.join(extracted_ingrediants).strip()
         print(st.session_state.extracted_ingrediants)

         doc = nlp(clean_text)

         html = displacy.render(doc)
         # Double newlines seem to mess with the rendering
         html = html.replace("\n\n", "\n")
         st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

         ents = doc.ents
         for ent in ents:
            scr = results_map[str(ent)]['scores'][0]
            new_label = f"{ent.label_} ({float(scr):.0%})"
            options["colors"][new_label] = options["colors"].get(ent.label_.lower(), None)
            options["ents"].append(new_label)
            ent.label_ = new_label
         doc.ents = ents

         html = displacy.render(doc, style="ent", options=options)
         # Double newlines seem to mess with the rendering
         html = html.replace("\n\n", "\n")
         st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

         cuisine = loaded_model.predict(loaded_tfidf_pip.fit_transform([st.session_state.extracted_ingrediants]))[0]
         cuisine_name = loaded_lb_encoder.classes_[cuisine]
         cluster_num = loaded_km_model.predict(loaded_km_tfidf_pip.fit_transform([f"{st.session_state.extracted_ingrediants}, {cuisine_name}"]))[0]
         filterd_df = loaded_dishes[loaded_dishes.cluster_num == cluster_num].sample(5)
         new_df = loaded_indDF[loaded_indDF.name.isin(filterd_df.name)]
         st.title("Welcome to  dine and ......")
         st.dataframe(new_df)
      
