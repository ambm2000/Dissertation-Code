
# Harassment Policy Analysis Codebase

This repository contains the complete code used for my MEng Computer Science dissertation titled:

**"Framing Harm: A Multi-Method Analysis of Online Harassment Policies Across Platform Types"**

## Research Questions
1. **RQ1**: What are the most prevalent topics in harassment-related policy documentation, and how do they differ across platform types?
2. **RQ2**: How do online platforms differ in their framing and discussion of harassment-related harms in reporting policies?

## Project Structure

### Topic Modelling (RQ1)
- `RQ1unsupervised_topic_modelling.py`  
- `RQ1unsupervised_topic_LLM.py`  
- `RQ1unsupervised_bertopic_visual.py`  
- `RQ1guided_topic_modelling.py`  
- `RQ1guided_topic_LLM.py`  
- `RQ1guided_bertopic_visual.py`  
- `RQ1LLM_tagged.py`  

### Harm Typing & Framing Analysis (RQ2)
- `RQ2harm_typing.py`  
- `RQ2harm_cluster_visual.py`  
- `RQ2harm_stats_analysis.py`  
- `RQ2sentiment_scoring.py`  
- `RQ2platform_type_comparison.py`  
- `RQ2step4_harm_analysis.py`  
- `RQ2clusteringBERT.py`  

### Preprocessing
- `extract_text.py`  
- `preprocess_texts.py`  
- `process_missing_files.py`  
- `requirements.txt`

### Output Directories
- `bertopic_results_unsupervised/`  
- `bertopic_results_guided/`  
- `bertopic_results_unsupervised_visual/`  
- `bertopic_results_guided_visual/`  
- `RQ2output/`

## Notes
- All code is written in Python and relies on standard NLP/data science libraries including BERTopic, HuggingFace Transformers, pandas, and scikit-learn.
- See `requirements.txt` for the list of dependencies.

## Contact
For any questions, or for the dataset files feel free to reach out via GitHub issues or email me at zcabbin@ucl.ac.uk.
