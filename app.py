import os

import streamlit as st
from nltk.tokenize import sent_tokenize
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List


COLOUR_MAP = {
    "Goal": "#F57C00",  # dark orange
    "Fact": "#4CAF50",  # medium green
    "Result": "#2196F3",  # blue
    "Hypothesis": "#9FA8DA",  # desaturated blue
    "Method": "#FFC107",  # amber
    "Problem": "#AB47BC",  # violet
    "Implication": "#EF5350",  # red
    "None": "#BDBDBD",  # medium grey
}


@st.cache_resource
def load_model(provider_choice: str, model_choice: str, _api_key: str):
    if provider_choice == "OpenAI":
        from langchain.chat_models import ChatOpenAI

        os.environ["OPENAI_API_KEY"] = _api_key
        llm = ChatOpenAI(model=model_choice, temperature=0.0, max_tokens=32)
    else:
        from langchain.llms import Cohere

        llm = Cohere(model=model_choice, cohere_api_key=_api_key, temperature=0.0)

    return llm


@st.cache_data
def run_chain(
    provider_choice: str, model_choice: str, _api_key: str, input_sentences: List[str]
):
    llm = load_model(
        provider_choice=provider_choice, model_choice=model_choice, _api_key=_api_key
    )
    prompt = PromptTemplate(
        input_variables=["input_sentences"],
        template="""You are a scientific discourse tagging bot. You will be given one or more consecutive sentences from a scientific paper and should classify each into one of the following types:

Type,Description
Goal,Research goal
Fact,A known fact, a statement taken to be true by the author
Result,The outcome of an experiment
Hypothesis,A claim proposed by the author
Method,Experimental method
Problem,An unresolved or contradictory issue
Implication,An interpretation of the results

Your answer should be a comma-separated list of types (and nothing else) for each given input sentence. E.g. for a five-sentence input, you might output: "Goal, Fact, Result, None, Implication". Types MUST come from the list above. If you can't identify a suitable type, use "None". 

{input_sentences}

Output:""",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(input_sentences=input_sentences)
    return output


def main():
    st.title("Scientific Discourse Tagging 🔬 💬 🏷️")
    st.subheader(
        "An experiment in scientific discourse tagging using few-shot in-context learning (ICL)"
    )
    st.write(
        """
        Scientific discourse tagging involves labelling clauses or sentences in a scientific paper with distinct rhetorical elements.
        This task is valuable for [scholarly document processing (SDP)](https://ornlcda.github.io/SDProc/) as it enables us (for example) to automatically distinguish
        between _observations_ made in experiments and their _implications_, as well as to differentiate between _claims_ backed by
        evidence and _hypotheses_ proposed to prompt further research ([Xiangci Li, Gully Burns, and Nanyun Peng, 2021](https://aclanthology.org/2021.eacl-main.218/)).
        This demo is an experiment in scientific discourse tagging using few-shot in-context learning (ICL).
        """
    )

    st.subheader("How to use this demo ❓")

    st.write(
        """
        Enter a sentence or paragraph of text from a scientific paper (or choose one of the examples). Each sentence will be classified as one of the following types:
        
- __Goal__: Research goal
- __Fact__: A known fact, a statement taken to be true by the author
- __Result__: The outcome of an experiment
- __Hypothesis__: A claim proposed by the author
- __Method__: Experimental method
- __Problem__: An unresolved or contradictory issue
- __Implication__: An interpretation of the results

"""
    )
    st.caption(
        "Discourse tags come from ([de Waard & Pander Maat, 2012](https://aclanthology.org/W12-4306/))"
    )

    with st.sidebar:
        st.header("Settings")
        provider_choice = st.selectbox(
            "Choose a model:",
            ["OpenAI", "Cohere"],
            index=0,
            help="Choose a model provider. At the time of writing, OpenAI models perform the best at this task."
            "",
        )
        model_choice = st.text_input(
            "Model name:",
            value="gpt-3.5-turbo"
            if provider_choice == "OpenAI"
            else "command-xlarge-nightly",
            help="Any valid model name for the chosen provider. Reasonable defaults are used.",
        )
        api_key = st.text_input(
            "Enter your API Key:", help="Your API key for the chosen provider."
        )

    if api_key:
        input_text = st.text_area(
            "Enter some text here",
            disabled=False,
            placeholder="Sentence or paragraph of text from a scientific paper...",
            help="test",
        )

        st.write("Or, choose one of the examples below:")

        # Three horizontal buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            example_biomedical = st.button("Example 1 (Biomedical)")
        with col2:
            example_compsci = st.button("Example 2 (CompSci)")
        with col3:
            example_physics = st.button("Example 3 (Physics)")
        with col4:
            example_chemistry = st.button("Example 4 (Chemistry)")

        if example_biomedical:
            input_text = "For unknown reasons, the melanocyte stem cell (McSC) system fails earlier than other adult stem cell populations1, which leads to hair greying in most humans and mice2,3. Current dogma states that McSCs are reserved in an undifferentiated state in the hair follicle niche, physically segregated from differentiated progeny that migrate away following cues of regenerative stimuli4-8. Here we show that most McSCs toggle between transit-amplifying and stem cell states for both self-renewal and generation of mature progeny, a mechanism fundamentally distinct from those of other self-renewing systems. Live imaging and single-cell RNA sequencing revealed that McSCs are mobile, translocating between hair follicle stem cell and transit-amplifying compartments where they reversibly enter distinct differentiation states governed by local microenvironmental cues (for example, WNT). Long-term lineage tracing demonstrated that the McSC system is maintained by reverted McSCs rather than by reserved stem cells inherently exempt from reversible changes. During ageing, there is accumulation of stranded McSCs that do not contribute to the regeneration of melanocyte progeny. These results identify a new model whereby dedifferentiation is integral to homeostatic stem cell maintenance and suggest that modulating McSC mobility may represent a new approach for the prevention of hair greying."
            st.write(
                "##### Example 1: [Dedifferentiation maintains melanocyte stem cells in a dynamic niche](https://pubmed.ncbi.nlm.nih.gov/37076619/)"
            )
        elif example_compsci:
            input_text = "We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4's performance based on models trained with no more than 1/1,000th the compute of GPT-4."
            st.write(
                "##### Example 2: [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)"
            )
        elif example_physics:
            input_text = "Our Moon periodically moves through the magnetic tail of the Earth that contains terrestrial ions of hydrogen and oxygen. A possible density contrast might have been discovered that could be consistent with the presence of water phase of potential terrestrial origin. Using novel gravity aspects (descriptors) derived from harmonic potential coefficients of gravity field of the Moon, we discovered gravity strike angle anomalies that point to water phase locations in the polar regions of the Moon. Our analysis suggests that impact cratering processes were responsible for specific pore space network that were subsequently filled with the water phase filling volumes of permafrost in the lunar subsurface. In this work, we suggest the accumulation of up to ~ 3000 km3 of terrestrial water phase (Earth’s atmospheric escape) now filling the pore spaced regolith, portion of which is distributed along impact zones of the polar regions of the Moon. These unique locations serve as potential resource utilization sites for future landing exploration and habitats (e.g., NASA Artemis Plan objectives)."
            st.write(
                "##### Example 3: [Distribution of water phase near the poles of the Moon from gravity aspects](https://www.nature.com/articles/s41598-022-08305-x)"
            )
        elif example_chemistry:
            input_text = "Given the increasing consumer demand for raw, nonprocessed, safe, and long shelf-life fish and seafood products, research concerning the application of natural antimicrobials as alternatives to preservatives is of great interest. The aim of the following paper was to evaluate the effect of essential oils (EOs) from black pepper (BPEO) and tarragon (TEO), and their bioactive compounds: limonene (LIM), β-caryophyllene (CAR), methyl eugenol (ME), and β-phellandrene (PHE) on the lipolytic activity and type II secretion system (T2SS) of Pseudomonas psychrophila KM02 (KM02) fish isolates grown in vitro and in fish model conditions. Spectrophotometric analysis with the p-NPP reagent showed inhibition of lipolysis from 11 to 46%. These results were confirmed by RT-qPCR, as the expression levels of lipA, lipB, and genes encoding T2SS were also considerably decreased. The supplementation of marinade with BPEO and TEO contributed to KM02 growth inhibition during vacuum packaging of salmon fillets relative to control samples. Whole-genome sequencing (WGS) provided insight into the spoilage potential of KM02, proving its importance as a spoilage microorganism whose metabolic activity should be inhibited to maintain the quality and safety of fresh fish in the food market."
            st.write(
                "##### Example 4: [Black pepper and tarragon essential oils suppress the lipolytic potential and the type II secretion system of P. psychrophila KM02](https://www.nature.com/articles/s41598-022-09311-9)"
            )

        if input_text.strip():
            sentences = sent_tokenize(input_text)

            input_sentences = f"\n\n".join(
                f"Input sentence {i+1}: {sent}" for i, sent in enumerate(sentences)
            )
            output = run_chain(
                provider_choice,
                model_choice,
                _api_key=api_key,
                input_sentences=input_sentences,
            )
            tags = [tag.strip() for tag in output.strip().split(",")]

            formatted_text = ""
            for i, sent in enumerate(sentences):
                tag = tags[i]
                colour = COLOUR_MAP.get(tag, "None")
                formatted_text += f'<div style="color: {colour};">{sent.strip()} <strong>[{tag.upper()}]</strong></div>'

            # Use st.markdown to render the HTML and CSS
            st.write("##### Model output")
            st.markdown(formatted_text, unsafe_allow_html=True)
    else:
        input_text = st.text_area("Enter your text here", value="", disabled=True)
        st.warning(
            "Please enter your API key in the sidebar to enable the input text box and examples."
        )

    "---"
    st.subheader("Observations 🧐")
    st.write(
        """
        - OpenAI models currently perform the best. I was not able to solve the task at all with Cohere models.
        - `gpt.3.5-turbo` performs reasonably well, I did not try `gpt-4` but assume it would perform better.
        - The model appears to perform better on longer texts. Performance on short texts is not great.
        - The model appears to perform the best for biomedical texts. It also performs well for the natural sciences (e.g. physics), but not as well for computer science.
        - All examples are scientific papers published _after_ GPT-3/3.5/4's data cutoff date of September 2021.
        - I am simply using NLTK for sentence tokenization, which is likely not optimal. Some sentences clearly belong to multiple tags.
        """
    )

    st.subheader("Feedback 📝")
    st.write(
        "Feel free to leave an issue or open a pull request on [GitHub](https://github.com/JohnGiorgi/SciTagger)!"
    )


if __name__ == "__main__":
    main()
