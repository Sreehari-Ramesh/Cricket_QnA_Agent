import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd


# List of DBs with their configs
DATABASES = {
    "iplcricketdb.db": {
        "vector_store": "ipl_vector_store",
        "teams_csv": "ipl_teams.csv",
        "team_code_col": "Abbreviation",
        "team_name_col": "Team",
    },
    "t20icricketdb.db": {
        "vector_store": "t20i_vector_store",
        "teams_csv": "icc_members.csv",
        "team_code_col": "Code",
        "team_name_col": "Country",
    },
    "pslcricketdb.db": {
        "vector_store": "psl_vector_store",
        "teams_csv": "psl_teams.csv",
        "team_code_col": "Abbreviation",
        "team_name_col": "Team",
    },
    "odicricketdb.db": {
        "vector_store": "odi_vector_store",
        "teams_csv": "icc_members.csv",
        "team_code_col": "Code",
        "team_name_col": "Country",
    },
    "bplcricketdb.db": {
        "vector_store": "bpl_vector_store",
        "teams_csv": "bpl_teams.csv",
        "team_code_col": "Abbreviation",
        "team_name_col": "Team",
    },
    "ilt20cricketdb.db": {
        "vector_store": "ilt20_vector_store",
        "teams_csv": "ilt_teams.csv",
        "team_code_col": "Abbreviation",
        "team_name_col": "Team",
    },    
    "bblcricketdb.db": {
        "vector_store": "bbl_vector_store",
        "teams_csv": "bbl_teams.csv",
        "team_code_col": "Abbreviation",
        "team_name_col": "Team",
    },
    "cplcricketdb.db": {
        "vector_store": "cpl_vector_store",
        "teams_csv": "cpl_teams.csv",
        "team_code_col": "Abbreviation",
        "team_name_col": "Team",
    },
    "t20bcricketdb.db": {
        "vector_store": "ntb_vector_store",
        "teams_csv": "t20b_teams.csv",
        "team_code_col": "Abbreviation",
        "team_name_col": "Team",
    },
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_entities_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # --- Players ---
    cur.execute("SELECT player_full_name, player_shortform FROM player_bio")
    players_raw = cur.fetchall()
    players_list = []
    for full, short in players_raw:
        if full:
            combined_name = f"{full} {short}" if short else full
            players_list.append(combined_name.strip())

    # --- Teams ---
    cur.execute("SELECT DISTINCT team1, team2 FROM matches")
    teams_raw = cur.fetchall()
    teams_set = set()
    for t1, t2 in teams_raw:
        if t1:
            teams_set.add(t1.strip())
        if t2:
            teams_set.add(t2.strip())
    teams_list = list(teams_set)

    # --- Tournaments ---
    cur.execute("SELECT DISTINCT tournament_name FROM tournaments")
    tournaments_list = [row[0].strip() for row in cur.fetchall() if row[0]]

    # --- Venues ---
    cur.execute("SELECT DISTINCT venue FROM matches")
    venues_list = [row[0].strip() for row in cur.fetchall() if row[0]]

    conn.close()
    return players_list, teams_list, tournaments_list, venues_list


def create_vector_store(docs, store_name):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(f"{store_name}/faiss_index")
    print(f"✅ Vector store created: {store_name}")


if __name__ == "__main__":
    for db_path, config in DATABASES.items():
        print(f"\n🚀 Processing {db_path} ...")
        players, teams, tournaments, venues = extract_entities_from_db(db_path)

        # Build team map from CSV
        df = pd.read_csv(config["teams_csv"])
        team_map = {}
        for _, row in df.iterrows():
            if row[config["team_code_col"]]:
                team_map[row[config["team_name_col"]].strip().lower()] = row[config["team_code_col"]].strip().upper()

        # Build docs
        docs = []
        for name in players:
            docs.append(Document(page_content=name, metadata={"type": "player"}))

        for name in teams:
            key = name.lower().strip()
            abbrev = team_map.get(key, None)
            combined = f"{name} ({abbrev})" if abbrev else name
            docs.append(Document(page_content=combined, metadata={"type": "team"}))

        for name in tournaments:
            docs.append(Document(page_content=name, metadata={"type": "tournament"}))

        for name in venues:
            docs.append(Document(page_content=name, metadata={"type": "venue"}))

        # Create vector store
        create_vector_store(docs, config["vector_store"])


    # retrieved_data = embedding_retrieval({
    # "team": ["LQ"]})
    # print("Retrieved Documents:",retrieved_data)

