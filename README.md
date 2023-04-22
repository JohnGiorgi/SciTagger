# SciTagger

Scientific discourse tagging involves labelling clauses or sentences in a scientific paper with distinct rhetorical elements. This task is valuable for [scholarly document processing (SDP)](https://ornlcda.github.io/SDProc/) as it enables us (for example) to automatically distinguish between _observations_ made in experiments and their _implications_, as well as to differentiate between _claims_ backed by evidence and _hypotheses_ proposed to prompt further research ([Xiangci Li, Gully Burns, and Nanyun Peng, 2021](https://aclanthology.org/2021.eacl-main.218/)). This demo is an experiment in scientific discourse tagging using few-shot in-context learning (ICL).

See the demo for more details [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/johngiorgi/SciTagger/main/app.py). And see [`app.py`](app.py) for the code and model prompt.

## Running locally

To run locally, you can either install from GitHub

```bash
pip install git+https://github.com/JohnGiorgi/SciTagger.git
pip install streamlit
```

or clone the repo and install from source using [Poetry](https://python-poetry.org/)

```bash
git clone https://github.com/JohnGiorgi/SciTagger.git
cd SciTagger
poetry install
```

Then run the demo

```bash
poetry run streamlit run app.py
```
