# compliance-agent-fbc
compliance agent, fbc ai policy, fbc security principles, NIST 2.0 (800-53), SOC 2



1. download ollama

2. in your terminal we have to get qwen2.5:3b, paste this command into your terminal:

ollama run qwen2.5:3b


3. paste the following into your terminal in visual studio code (llangchian, streamlit)

pip install streamlit langchain langchain-chroma langchain-ollama langchain-community sentence-transformers

4. after installing the necessary extensions, clone this git hub repo


5. after cloning the repo run this on your terminal below:

python3 build_vault.py

python3 -m streamlit run app.py


**Labeling Guidelines:**
* MET = The procedure explicitly addresses the core requirement of the control.
* PARTIAL = The procedure performs the core activity but omits a required element, or only applies to a subset of systems.
* NOT_MET = The activity is completely absent, or the text explicitly contradicts the requirement.
* NOT_APPLICABLE = The text describes a procedure completely unrelated to the control's domain (e.g., door locks vs. patching).





