BAIT (BITS AI Tutor) is an AI tool for last minute preparation for mid and end-semester exam preparation. 
It utilises Retrieval-Augmented Generation(RAG) with multimodal inputs to provide tailored learning experience. 
We currently focus on humanities courses offered at BITS Pilani, with plans to expand its capabilities to a wider range of courses in future.

Folder Structure
Project-BAIT/
│
├── app.py
├── requirements.txt
├── README.md
│
├── courses/
│   ├── Professional Ethics/
│   │   ├── topics.json
│   │   ├── question_bank.json
│   │   ├── faiss_store/
│   │   │   ├── index.faiss
│   │   │   ├── index.pkl
│   │   │   └── (any metadata files)
(similar for other courses)

