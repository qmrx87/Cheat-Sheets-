  **20/03/2025**  

> **Higher School Of Computer Science - Sidi Bel Abbès**  
> **DAIT DEHANE Yacine**  
> **Natural Language Processing Group**

---

## Outline

| Phase | Topics |
|-------|--------|
| **Data Preparation** | Pretraining Data → Tokenization → Transformer |
| **LLM Training** | Pre-training → Instruction Tuning → RLHF |
| **LLM Usage** | Evaluation → Prompt Engineering → Large Reasoning Models |
| **Applications** | **Agent → Multi-Agent → Real-World Use Cases** |
| **Conclusion** | Summary & Future Directions |

---

# Hands-on Applications of Large Language Models

Below is a **comprehensive, practical cheat sheet** focused on **real-world applications** of LLMs — from building agents to enterprise deployment.

---

## 1. LLM-Powered Agents

> **Definition**: Autonomous systems that use LLMs to **perceive**, **plan**, **act**, and **reflect** in dynamic environments.

### Core Components
```text
[Observation] → [LLM Reasoning] → [Action] → [Environment] → [New Observation]
```

| Component | Tool / Framework | Example |
|---------|------------------|-------|
| **Memory** | LangChain Memory, LlamaIndex | Short-term (chat), Long-term (vector DB) |
| **Tools** | LangChain Tools, OpenAI Function Calling | Web search, code execution, API calls |
| **Planning** | ReAct, Reflexion, Tree-of-Thought | Step-by-step reasoning |
| **Execution** | AutoGPT, BabyAGI, MetaGPT | Autonomous task completion |

### Hands-on Example: **ReAct Agent** (Reason + Act)

```python
from langchain import OpenAI, PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun

llm = OpenAI(temperature=0)
tools = [Tool(name="Search", func=DuckDuckGoSearchRun(), description="Useful for searching the web.")]

agent = initialize_agent(tools, llm, agent="react-description", verbose=True)

agent.run("What is the latest version of Python and its release date?")
```

**Output trace**:
```
Thought: I need to search for the latest Python version.
Action: Search
Action Input: "latest python version release date"
Observation: Python 3.13.0 was released on October 7, 2024.
Thought: I now know the final answer.
Final Answer: The latest version of Python is 3.13.0, released on October 7, 2024.
```

---

## 2. Multi-Agent Systems

> Multiple specialized agents collaborate to solve complex tasks.

| Framework | Use Case | Key Feature |
|---------|--------|-----------|
| **MetaGPT** | Software Company Simulation | CEO → PM → Engineer → QA |
| **CAMEL** | Role-Playing Agents | Inception prompting |
| **AutoGen (Microsoft)** | Conversational Agents | Customizable agent roles |

### Example: **MetaGPT – Build a CLI Game**

```bash
pip install metagpt
```

```python
from metagpt.software_company import SoftwareCompany

company = SoftwareCompany()
company.invest(idea="A CLI-based Snake game in Python")
company.hire_roles()
company.run_project()
```

**Output**: Generates `snake.py`, `requirements.txt`, `README.md`, tests.

---

## 3. Prompt Engineering in Practice

| Technique | Template | Use Case |
|---------|--------|--------|
| **Zero-Shot** | `"Translate to French: Hello world"` | Simple tasks |
| **Few-Shot** | `"English: I love coding\nFrench: J'adore programmer\nEnglish: Good night\nFrench:"` | In-context learning |
| **Chain-of-Thought (CoT)** | `"Solve step by step: ..."` | Math, logic |
| **Self-Consistency** | Generate 5 answers → majority vote | Robust reasoning |
| **Tree of Thoughts (ToT)** | Explore multiple paths | Planning, search |
| **ReAct** | `"Thought: ...\nAction: ...\nObservation: ..."` | Tool use |

### CoT Prompt Example

```text
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many does he have now?
A: Let's solve step by step.
Roger started with 5 balls.
2 cans × 3 balls/can = 6 balls.
5 + 6 = 11.
Answer: 11
```

---

## 4. Evaluation Beyond Benchmarks

| Metric | Tool | Purpose |
|------|------|--------|
| **Perplexity** | `transformers` | Language modeling quality |
| **BLEU/ROUGE** | `sacrebleu`, `rouge-score` | Translation, summarization |
| **HumanEval** | OpenAI Eval | Code generation |
| **HELM / BigBench** | Holistic eval suites | General capability |
| **Reward Models** | OpenAI RLHF | Preference alignment |

### Run HumanEval Locally

```bash
pip install human-eval
evaluate_functional_correctness sample-input.jsonl sample-output.jsonl
```

---

## 5. Real-World Deployment

| Use Case | Model | Deployment |
|--------|-------|-----------|
| **Customer Support** | Llama 3, Mistral | RAG + Function Calling |
| **Legal Document Review** | Claude 3, GPT-4 | Fine-tuned + RAG |
| **Code Assistant** | CodeLlama, DeepSeek-Coder | VS Code Plugin |
| **Education Tutor** | Phi-3, Gemma | Adaptive learning paths |
| **Healthcare Assistant** | Med-PaLM, BioGPT | HIPAA-compliant RAG |

### RAG Pipeline (Retrieval-Augmented Generation)

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 1. Load docs
docs = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(docs)

# 3. Embed + Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Retrieve + Generate
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
qa.run("What is the refund policy?")
```

---

## 6. Fine-Tuning LLMs (Hands-on)

### Using **Hugging Face + PEFT + LoRA**

```bash
pip install peft transformers datasets accelerate bitsandbytes
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_name = "meta-llama/Llama-3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", load_in_8bit=True
)

# LoRA Config
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)

# Train on custom dataset (Alpaca format)
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="llama3-lora-finetuned"
    )
)
trainer.train()
```

---

## 7. RLHF in Practice (Simplified)

```python
# Use trl library (from Hugging Face)
from trl import PPOTrainer, PPOConfig

ppo_trainer = PPOTrainer(
    model=model,
    config=PPOConfig(batch_size=16),
    tokenizer=tokenizer
)

# Generate responses → get human feedback → train reward model → PPO update
```

---

## 8. Safety & Alignment

| Technique | Tool |
|---------|------|
| **Constitutional AI** | Anthropic Claude |
| **Red Teaming** | `garak`, `scale-red-team` |
| **Content Filters** | `moderation-api`, `perspective-api` |
| **Guardrails** | `NVIDIA NeMo Guardrails` |

### Example: Guardrails

```yaml
flows:
  - name: "no_pii"
    steps:
      - action: "self_check_input"
        next: END
      - action: "generate"
```

---

## 9. Tools & Ecosystem (2025)

| Category | Top Tools |
|--------|----------|
| **Frameworks** | LangChain, LlamaIndex, Haystack, AutoGen |
| **Model Hubs** | Hugging Face, Ollama, vLLM, TensorRT-LLM |
| **Inference** | vLLM, TGI, Ollama, LM Studio |
| **Agents** | AutoGPT, BabyAGI, MetaGPT, CrewAI |
| **Evaluation** | HELM, Open LLM Leaderboard, LM-Eval-Harness |
| **MLOps** | MLflow, Weights & Biases, TruLens |

---

## 10. Future Directions (Hands-on Ready)

| Trend | How to Experiment Today |
|------|-------------------------|
| **Multimodal Agents** | LLaVA + LangChain |
| **Long Context (1M+)** | Gemini 1.5, Infini-Attention |
| **On-Device LLMs** | MLX, Ollama, Phi-3 Mini |
| **Agent-to-Agent Comms** | AutoGen Studio |
| **LLM + Robotics** | RT-2, PALM-E |


Below is the **new section** you can copy-paste into your cheat-sheet, followed by the **exact location** where to inject it **without rewriting the whole file**.

---

## New Section: Limitations of LLMs & Mitigation Strategies

```markdown
## 11. Limitations of LLMs & How to Address Them

| Limitation | Description | Real-World Impact | Mitigation |
|----------|-------------|-------------------|----------|
| **Outdated Knowledge** | Trained on fixed data → no real-time updates | Wrong facts after cutoff (e.g., “Who won the 2025 election?”) | **RAG (Retrieval-Augmented Generation)**: Fetch latest docs from DB/API |
| **Hallucinations** | Generates plausible but false info | Misleading answers in legal/medical use | **RAG + Verification**, **Self-Consistency**, **Fact-Checking Tools** |
| **Lack of Domain Expertise** | Weak in niche fields (law, medicine, finance) | Poor accuracy where precision is critical | **Domain-Specific Fine-Tuning**, **RAG with expert corpus** |
| **Poor Explainability** | “Black box” reasoning | Not acceptable in regulated domains | **Agentic RAG**, **GraphRAG**, **Chain-of-Thought Tracing** |
| **Long-Context Blindness** | Forgets earlier parts of input | Inconsistent in long docs or chats | **GraphRAG**, **Memory-Augmented Agents**, **Sliding Window + Summary** |
| **Bias & Fairness Issues** | Reflects training data biases | Discriminatory outputs | **Debiasing Prompts**, **Constitutional AI**, **Fairness Audits** |

---

### Key Mitigation Techniques (Hands-on)

#### 1. **RAG** – Retrieve & Generate
```python
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
qa.run("Latest AI regulations in EU?")
```

#### 2. **Agentic RAG** – Retrieve + Reason + Act
```python
from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools=[search_tool, db_tool])
agent.run("Update me on GDPR changes in 2025.")
```

#### 3. **GraphRAG** – Knowledge Graphs for Structure
```python
from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
graph.query("MATCH (n:Regulation)-[:APPLIES_TO]->(c:Country {name:'EU'}) RETURN n")
```
> Use **Microsoft GraphRAG** or **LlamaIndex Knowledge Graphs** to connect entities and reduce hallucinations.

---

**Pro Tip**: Combine **RAG → Agent → Graph** for **enterprise-grade reliability**.

```

 


```bash
# Run local LLM
ollama run llama3

# Start agent
autogpt --demo

# Fine-tune with LoRA
python lora_finetune.py --model meta-llama/Llama-3-8b

# Deploy RAG API
uvicorn rag_api:app --reload
```

---



## Resources

| Type | Link |
|------|------|
| **Book** | [Hands-on Large Language Models](https://github.com/xiachong/hands-on-llms) |
| **Course** | [CS224N – Stanford NLP](https://web.stanford.edu/class/cs224n/) |
| **Leaderboard** | [Hugging Face Open LLM](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) |
| **Agents** | [AutoGen Docs](https://microsoft.github.io/autogen/) |
| **RAG** | [LlamaIndex Tutorials](https://gpt-index.readthedocs.io/) |

---

## Conclusion

> **LLMs are not just models — they are programmable intelligence engines.**  
> With the right **data**, **tools**, and **prompting**, you can build **autonomous agents**, **enterprise copilots**, and **next-gen AI applications**.

---

**Ready to build?**  
Start with **FineWeb + Llama 3 + LangChain + ReAct** → deploy your first agent in **<100 lines**.

---
 

--- 

 