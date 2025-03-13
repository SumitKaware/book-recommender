import pandas as pd
import numpy as np
#import dotenv import load_dotenv
import gradio as gr
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

#load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&file=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
print("Document Loaded")
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
print("Chunks created")
documents = text_splitter.split_documents(raw_documents)
print("Text Splitted")
#db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings())
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)
print("Vector DB created")

def retrive_semantic_recommendation(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k:  int = 50,
        final_top_k: int = 10
) -> pd.DataFrame:
    print("Inside retrive_semantic_recommendation")
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0] for rec in recs)]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy",ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprie",ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="angry",ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear",ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness",ascending=False, inplace=True)
    
    return book_recs

print("retrive_semantic_recommendation Ended")

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    print("Inside recommend_books")
    recommendation = retrive_semantic_recommendation(query, category, tone)
    results = []

    for _, row in recommendation.iterrows():
        description = row["description"]
        truncated_decs_split = description.split()
        truncated_description = " ".join(truncated_decs_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append(row["large_thumbnail"], caption)
    
    return results

print("recommend_books Ended")

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
        
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter the description",
                                placeholder="ex: lovestory")
        category_dropdown = gr.Dropdown(choices=categories, label="Select Category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select tone:", value="All")
        submit_button = gr.Button("Show Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)
    

if __name__ == "__main__":
    dashboard.launch()
        