# RAG for Cricket Query System

This repository contains the **Retriever–Augmented Geneation (RAG)** workflows for answering cricket-related queries for **T20 International (T20I)** , **Indian Premier League (IPL)** datasets and **Pakistan Super League (PSL)**

We have three main RAG pipelines:

* **RAG-T20** → For answering queries on T20 International matches
* **RAG-IPL** → For answering queries on IPL matches
* **RAG-PSL** → For answering queries on PSL matches

Both pipelines use a **vector store** created from structured cricket data to enable semantic retrieval and natural language querying.

---

## 📂 Workflow Overview

### 1. Entity Extraction

* User queries are first parsed to extract cricket-specific entities such as:

  * **Players** (full names / short names)
  * **Teams**
  * **Tournaments**
  * **Venues**
  * **Years/Seasons**
* This ensures that queries are contextually mapped to the right dataset.

### 2. Retriever-Augmented Generation (RAG)

We maintain **separate RAG workflows** for T20I and IPL:

* **RAG-T20**:

  * Uses the **T20I database tables**.
  * Retrieves relevant data (batting, bowling, match summaries, head-to-head, etc.).
  * Constructs SQL queries dynamically based on extracted entities.

* **RAG-IPL**:

  * Uses the **IPL database tables**.
  * Similar process, but adapted for IPL-specific tables like franchise stats, season summaries, etc.

* **RAG-PSL**:

  * Uses the **PSL database tables**.
  * Similar process, but adapted for PSL-specific tables like franchise stats, season summaries, etc.

Both RAGs follow the same pattern:

1. **Receive query**
2. **Resolve entities**
3. **Fetch from vector store & database**
4. **Generate final natural language response**

### 3. Vector Store Creation

* We use **FAISS** as the backend for semantic retrieval.
* The vector store is built on top of structured descriptions of the cricket tables.
* This allows for **hybrid retrieval**:

  * **Exact match** → using resolved entities
  * **Semantic match** → using embeddings for related context

---

## ▶️ Running the Workflow

### Prerequisites

* Python 3.10+
* Required libraries:

  ```bash
  pip install -r requirements.txt
  ```

### Step 1: Create Vector Store

Run the vector store creation script (works for all the dbs we have):
Paths are also given in the code for seperate execution.

This will:

* Read schema/table descriptions from the database
* Embed them using `sentence-transformers`
* Store them in a FAISS local index

### Step 2: Run main agent:
* run the main_rag_agent to dynamically get the response from different databases we have.

---

## 💡 Example Usage

```text
> Who has the highest batting average in IPL 2020?  
→ RAG-IPL retrieves player batting stats for IPL 2020 and returns answer.  

> What is Virat Kohli’s strike rate in T20Is against Australia?  
→ RAG-T20 fetches player_vs_team stats and generates response.  
```
sample queries for testing:
```text
"who won the last ipl?"
"who was the top run scorer in ipl 2023?"
"who was the top run scorer in IPL 2022? what was his highest score?"
"who took most wickets in ipl 2025? and which team was him?"
"Which team scored the least runs in the ipl history and which season was it?"
"Which two teams played the final of the IPL 2021?"
"Which team scored the most runs in a single innings in whole IPL history?"
"who won the last ICC T20 World Cups?"
"who was the top run scorer in wolrdcup 2023?"
"who was the top run scorer in ICC worldcup 2022? what was his highest score?"
"who took most wickets in ICC T20 World Cup 2024?"
"Which team scored the least runs in the ICC T20 World Cup 2024?"
"Which two teams played the final of the ICC T20 World Cup 2021?"
"Which team scored the most runs in a single innings in World Cup history?"
"What is India's win/loss record against Australia in ICC tournaments?"
"Who scored the most runs in IPL 2023?"
"Top 2 wicket-takers in IPL 2020?"
"Which team won the IPL 2022 final and who was the Man of the Match?"
"Who is the leading run scorer in ICC world cup in 2022?"
"Compare Virat Kohli's best scores in IPL 2023 and ICC world cup 2022."
"Between Jos Buttler's IPL 2022 season and his T20 World Cup 2022 performance, which was better in terms of runs scored?"
"who won the last psl?"
"who was the top run scorer in psl 2023?"
"who was the top run scorer in psl 2022? what was his highest score?"
"who took most wickets in psl 2025? and which team was him?"
"Which team scored the least runs in the psl history and which season was it?"
"Which two teams played the final of the psl 2021?"
"Which team scored the most runs in a single innings in whole psl history?"
```
---

## 🔑 Key Notes

* You **do not** need to recreate the database (DB); it is already provided.
* Only the **RAG workflows and vector store creation** are required to run this system.
* Vector store improves retrieval by handling fuzzy queries and semantic context.
