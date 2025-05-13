InsightFlow AI - Project Deliverables Response
This document addresses the deliverables and questions based on the current implementation and planned features of the InsightFlow AI project.
TASK ONE – Defining your Problem and Audience
Q: Write a succinct 1-sentence description of the problem.
A: InsightFlow AI addresses the challenge of gaining deep, multi-faceted understanding of complex topics which are often oversimplified when viewed through a single lens or by using traditional research tools.
Q: Write 1-2 paragraphs on why this is a problem for your specific user.
A: Users like researchers, students, analysts, and decision-makers often struggle to obtain a holistic view of complex subjects. Standard search engines and many AI assistants tend to provide linear, singular answers, potentially missing crucial nuances, alternative viewpoints, or interdisciplinary connections. This can lead to incomplete understanding, confirmation bias, and suboptimal decision-making.
For instance, a student researching the "impact of AI on society" needs more than just a technical summary; they benefit from philosophical considerations of ethics, futuristic speculations on societal shifts, factual data on economic impact, and analytical breakdowns of different AI capabilities. Without a tool to easily access and synthesize these varied perspectives, users spend excessive time manually seeking out and trying to reconcile diverse information, or worse, they operate with an incomplete picture.
TASK TWO – Propose a Solution
Q: Propose a Solution (What is your proposed solution? Why is this the best solution? Write 1-2 paragraphs on your proposed solution. How will it look and feel to the user? Describe the tools you plan to use in each part of your stack. Write one sentence on why you made each tooling choice.)
A: Proposed Solution: InsightFlow AI is a multi-perspective research assistant that processes user queries through a configurable team of specialized AI "personas," each embodying a distinct reasoning style (e.g., Analytical, Scientific, Philosophical). It then synthesizes these diverse viewpoints into a single, coherent response, augmented by optional visualizations (DALL-E sketches, Mermaid concept maps) and Retrieval Augmented Generation (RAG) from persona-specific knowledge bases.
Why Best & User Experience: This approach is superior because it directly tackles the problem of narrow perspectives by design, offering a "team of experts" on demand. The user interacts via a clean Chainlit interface, configuring their AI team and desired output features (like RAG, visualizations, direct/quick modes) through an intuitive settings panel (⚙️). They ask a question and receive a rich, synthesized answer, with the option to explore individual persona contributions. The /help command provides guidance. This feels like having a dedicated, versatile research team at their fingertips.
Technology Stack:
LLM: OpenAI's GPT models (e.g., gpt-4o-mini, gpt-3.5-turbo) are used for their strong reasoning, generation, and instruction-following capabilities, powering persona responses and synthesis.
Embedding: sentence-transformers/all-MiniLM-L6-v2 (via langchain-huggingface) is the current default for RAG, chosen for its balance of performance and efficiency, with plans to use fine-tuned versions and an option for OpenAI's text-embedding-3-small.
Orchestration: LangGraph is used to define and manage the complex workflow of multi-persona query processing, RAG, synthesis, and visualization in a modular and clear way.
Vector Database: Qdrant (in-memory) is used for storing and querying embeddings for the RAG system, selected for its performance and scalability features, with langchain-qdrant for integration.
User Interface: Chainlit provides a rapid development framework for creating interactive chat applications with features like chat settings and custom message elements.
Visualization: DALL-E (via OpenAI API) for AI-generated image sketches and Mermaid.js (rendered by Chainlit) for concept maps provide visual aids to understanding.
Monitoring (Planned): LangSmith will be used for detailed tracing, debugging, and monitoring of LLM calls and LangGraph execution.
Evaluation (Planned): RAGAS will be employed to quantitatively evaluate the performance of the RAG pipeline, especially after fine-tuning embedding models.
Q: Where will you use an agent or agents? What will you use “agentic reasoning” for in your app?
A: Currently, the LangGraph orchestrates a defined flow. Agentic reasoning is planned for future enhancements:
The run_planner_agent node could become truly agentic, analyzing the query to dynamically select the most relevant personas or decide if external tools (like web search via Tavily) are needed before persona execution.
Individual execute_persona_tasks could evolve so each persona acts as a mini-agent, potentially with the ability to use specific tools (e.g., a "Scientific" persona using a calculator or data analysis tool) or to self-critique and refine its own output.
The RAG data retrieval step within execute_persona_tasks is an early form of agentic information gathering, as the system decides to fetch external knowledge.
TASK THREE – Dealing With the Data
Q: Describe all of your data sources and external APIs, and describe what you’ll use them for.
A: Data Sources for RAG:
Current: The RAG system is designed to load .txt files from persona-specific directories within data_sources/ (e.g., data_sources/analytical/, data_sources/philosophical/, data_sources/metaphorical/). These currently contain example texts or placeholder content relevant to the "Balanced Team" personas.
Planned (from Design Document): The full vision involves sourcing a wide array of texts for each of the six core personas and three personalities, including public domain literary works (e.g., Sherlock Holmes, Plato), scientific lectures (Feynman), news articles, and other domain-specific documents to build rich knowledge bases for each persona.
External APIs:
OpenAI API: Used for:
Powering the LLMs for persona response generation (ChatOpenAI).
The synthesis LLM (llm_synthesizer).
The direct mode LLM (llm_direct).
Generating DALL-E images for visualization.
(Potentially) Generating synthetic data for fine-tuning embeddings.
(If chosen as an option) Providing embeddings via OpenAIEmbeddings (e.g., text-embedding-3-small).
Hugging Face Hub (Implicit API): Used to download pre-trained SentenceTransformer models (like all-MiniLM-L6-v2) and will be used to host and download your fine-tuned embedding models.
Q: Describe the default chunking strategy that you will use. Why did you make this decision?
A: The current default chunking strategy, implemented in utils/rag_utils.py via load_and_split_documents, uses Langchain's RecursiveCharacterTextSplitter. It's configured with a chunk_size of 1000 characters and a chunk_overlap of 150.
Why: This strategy was chosen as a robust general-purpose starting point. RecursiveCharacterTextSplitter attempts to split text along semantic boundaries (paragraphs, then sentences, then words) before resorting to hard character limits. This helps maintain the coherence of the information within each chunk, which is crucial for effective retrieval and for the LLM to understand the context. The default sizes are common starting points, and the system is architected such that persona-specific chunking strategies could be implemented later if evaluation proves it necessary.
Q: [Optional] Will you need specific data for any other part of your application? If so, explain.
A: Yes, for fine-tuning the embedding model(s). As outlined in Fine_tuning_Balanced_Team_Embedding.ipynb, we will need curated datasets for each persona (or group of personas like the "Balanced Team") to create synthetic question-context pairs. This data will be drawn from the same sources intended for the RAG knowledge bases (e.g., texts for analytical, philosophical, metaphorical reasoning). The quality and specificity of this data are critical for successful fine-tuning.
TASK FOUR – Build a Quick End-to-End Prototype
Q: Build an end-to-end prototype and deploy it to a Hugging Face Space (or other endpoint).
A: An end-to-end prototype of InsightFlow AI has been built and is actively being developed.
Functionality: It features a Chainlit UI with a Chat Settings panel (⚙️) for configuring personas (individual or team-based selection like the default "Balanced Overview Group"), RAG enablement, display modes (Direct, Quick, show/hide perspectives & visualizations). It processes queries through a LangGraph pipeline involving planning, RAG context retrieval for enabled personas, parallel perspective generation, synthesis, and visualization (DALL-E & Mermaid). A /help command provides user guidance.
RAG Implementation: Basic RAG is functional, using all-MiniLM-L6-v2 embeddings by default (with an option to switch to OpenAI embeddings or a future fine-tuned model via UI toggle). It loads .txt files from data_sources/<persona_id>/ into in-memory Qdrant vector stores.
Deployment: The application is containerized using the provided Dockerfile which uses uv for dependency management from pyproject.toml. It is currently being tested for deployment on Hugging Face Spaces (target space: suh4s/InsightFlowAI_test). The README contains instructions and a link to the public GitHub repo. (The actual public deployment link and stability on HF Spaces is contingent on resolving any Git LFS issues and successful builds).
TASK FIVE – Creating a Golden Test Data Set
Q: Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevance, context precision, and context recall. Provide a table of your output results.
A: Status: Planned. The framework for RAGAS evaluation is understood, and the fine-tuning notebook (Fine_tuning_Balanced_Team_Embedding.ipynb) includes example RAGAS evaluation code.
Plan: Once the "Balanced Team" embedding model is fine-tuned using data from data_sources/analytical/, data_sources/philosophical/, and data_sources/metaphorical/, a golden test dataset will be curated. This dataset will consist of:
Test questions relevant to the Balanced Team's combined knowledge base.
Manually identified "ground truth" relevant document chunks for these questions.
Ideal answers generated based on these ground truth chunks.
The RAG pipeline (using both the default all-MiniLM-L6-v2 or OpenAI text-embedding-3-small, and the fine-tuned insightflow-balanced-team-embed-v1) will be run against these test questions.
RAGAS will then be used to calculate metrics such as faithfulness, answer_relevancy, context_precision, and context_recall.
(Currently, no RAGAS results table can be provided as this step is pending the fine-tuning and dataset creation).
Q: What conclusions can you draw about the performance and effectiveness of your pipeline with this information?
A: Anticipated Conclusions (Post-Evaluation): Based on the RAGAS evaluation, we anticipate being able to draw conclusions such as:
The effectiveness of the fine-tuned embedding model compared to the baseline general-purpose model for retrieving relevant context from the Balanced Team's specific knowledge base.
The impact of RAG on the faithfulness and relevance of the answers generated by the Analytical, Philosophical, and Metaphorical personas.
Identification of weaknesses in the retrieval process (e.g., low recall or precision for certain types of queries or documents) that can guide further improvements to chunking, embedding, or the RAG prompt.
Overall, whether the fine-tuned RAG integration leads to demonstrably better, more contextually grounded multi-perspective analyses for the Balanced Team.
TASK SIX – Fine-Tuning Open-Source Embeddings
Q: Swap out your existing embedding model for the new fine-tuned version. Provide a link to your fine-tuned embedding model on the Hugging Face Hub.
A: Current State & Plan:
The application currently defaults to OpenAI's text-embedding-3-small and has a UI toggle to switch to a fine-tuned model (placeholder ID: suh4s/insightflow-balanced-team-embed-v1).
The fine-tuning process for the insightflow-balanced-team-embed-v1 model (using data from data_sources/analytical/, data_sources/philosophical/, and data_sources/metaphorical/) will be conducted in Google Colab using the Fine_tuning_Balanced_Team_Embedding.ipynb notebook.
Once fine-tuned and evaluated, this model will be pushed to Hugging Face Hub under my username (e.g., suh4s/insightflow-balanced-team-embed-v1).
The FINETUNED_BALANCED_TEAM_EMBED_ID constant in app.py will be updated with this correct Hub ID.
Link (To Be Provided Post Fine-Tuning): The public link to the fine-tuned model on Hugging Face Hub will be provided here once the training and upload are complete. (Currently, this link is pending fine-tuning completion).
TASK SEVEN – Assessing Performance
Q: How does the performance compare to your original RAG application? Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements. Provide results in a table.
A: Planned Comparison:
The "original RAG application" will be defined as the RAG pipeline using the default embedding model (either all-MiniLM-L6-v2 if switched back, or text-embedding-3-small).
The "fine-tuned RAG application" will use the insightflow-balanced-team-embed-v1 model.
Both versions will be evaluated against the same golden test dataset created for the Balanced Team's knowledge domain using RAGAS.
A table comparing faithfulness, answer_relevancy, context_precision, and context_recall (and potentially other RAGAS metrics) for both the baseline and fine-tuned models will be generated.
(This comparative table and quantified improvements are pending the completion of fine-tuning and the RAGAS evaluation run).
Q: Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?
A: Expected Future Improvements:
Full RAG Data Sourcing: Complete the acquisition and processing of diverse, high-quality data sources for all six core personas and three personalities, building out their respective knowledge bases.
Persona-Specific Embedding Models: Fine-tune separate embedding models for each core persona (or logical groupings of personas) to maximize retrieval relevance for their unique domains and reasoning styles, beyond just the "Balanced Team" model.
Advanced RAG Techniques: Explore and implement techniques like re-ranking of retrieved documents, query transformation tailored to each persona, and hybrid search (keyword + semantic) if needed.
Agentic Enhancements: Make the run_planner_agent more intelligent to dynamically select personas and tools (like web search via Tavily) based on query analysis. Explore more sophisticated agentic interactions between personas if feasible.
Comprehensive Testing & Evaluation: Implement a more robust automated test suite (unit, integration tests). Systematically evaluate all persona RAG pipelines with RAGAS. Conduct user testing for qualitative feedback.
UI/UX Refinements: Further improve the Chainlit UI, potentially using Tabs for better organization of results, enhance the export functionality (currently stubbed), and refine the progress display.
Persistence: Implement session persistence and potentially allow users to save and manage their persona team configurations or knowledge bases.
Monitoring & Optimization: Integrate with LangSmith for robust monitoring and use insights to optimize prompts, LLM choices, and overall pipeline performance.
Deployment Hardening: Solidify the Hugging Face Spaces deployment, manage secrets securely, and ensure reliable operation.
Final Submission Links
Public GitHub Repo: https://github.com/suhas/AIE6_Cert_Challenge_Suhas (Contains all relevant code, this document, and other materials).
Loom Video Demo: (To be recorded and link inserted here)
Written Document (This Document): Included in the GitHub repository.
Public Application on Hugging Face: (Link to suh4s/InsightFlowAI_test Space on Hugging Face - will be active post-successful deployment)
Public Fine-tuned Embedding Model on Hugging Face: (Link to suh4s/insightflow-balanced-team-embed-v1 or similar on Hugging Face - will be active post fine-tuning and upload)