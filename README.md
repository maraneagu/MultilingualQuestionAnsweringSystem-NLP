# Multilingual Machine Reading — DIKU NLP Course Project (2025)

This project implements a full multilingual machine reading pipeline as part of the **Natural Language Processing** course at the University of Copenhagen (DIKU).  
The system works with the *TyDi QA*–derived multilingual dataset provided for the course and includes:

- **Rule-based classifiers**
- **Language modeling**
- **Binary answerability classification**
- **Token-level span extraction (sequence labeling)**
- **Multilingual answer generation (open QA)**

---

## Project Structure (Week-by-Week)

### **Week 36 — Dataset Exploration & Rule-Based Classifier**
- Explored the combined TyDi QA / XOR RC dataset  
- Extracted dataset statistics for Arabic, Korean, and Telugu  
- Identified the most frequent question words and analyzed linguistic patterns  
- Built and evaluated a **rule-based answerability classifier** using question–context overlap + optional machine translation
  
---

### **Week 37 — Language Modeling**
- Implemented **k different language models** (one per group member)  
- Trained language models for:
  - Arabic questions  
  - Korean questions  
  - Telugu questions  
  - English context documents  
- Compared n-gram and neural approaches and analyzed cross-lingual performance  

---

### **Week 38 — Learned Answerability Classifiers**
- Implemented **learned binary classifiers** for answerability prediction  
- Explored linguistic and embedding-based features:
  - Bag-of-words and TF–IDF  
  - N-gram statistics  
  - Multilingual sentence embeddings (mBERT / DistilBERT)  
- Evaluated models per language and analyzed cross-language differences  

---

### **Week 39 — Span Extraction (Sequence Labeling)**
- Trained **token-level sequence labelers** to extract answer spans from context documents  
- Evaluated using sequence labeling metrics  
- Compared multilingual vs. per-language training setups  
- Handled character-to-token index alignment as required by the dataset  

---

### **Week 40 — Open QA: Answer Generation**
Focused on **Telugu → English context → Telugu answer** generation using encoder–decoder models.

- Fine-tuned models such as **mT5-small**  
- Built multiple generation systems depending on group size:
  - Using Telugu question + English context  
  - Using only Telugu question  
  - Generating Telugu or English answers  
- Evaluated answer generation with text metrics  
- Compared performance on answerable vs. unanswerable questions  

---

### **Week 41+ — Custom Test Set & Final Evaluation**
- Created a **custom multilingual test set** with new questions, English contexts, and answer translations  
- Evaluated:
  - Best rule-based classifier  
  - Best learned answerability classifier  
  - Best span extraction model  
  - Best answer generation model  
- Discussed cross-model behavior and failure cases  

---

## Technologies & Tools
- **Python**, PyTorch, HuggingFace Transformers  
- **mT5, mBERT, DistilBERT**, n-gram language models  
- **HuggingFace Datasets** (TyDi QA / XOR RC)  
- Google Colab GPU environment  

---

## Dataset
This project uses the DIKU-provided multilingual dataset based on:
- **TyDi QA** (Clark et al., 2020)  
- **XOR RC** (Asai et al., 2021)  
- **XOR-AttriQA** (Muller et al., 2023)  

Dataset link: https://huggingface.co/datasets/coastalcph/tydi_xor_rc
