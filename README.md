---
title: GOSIM Agent
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
license: mit
---

## Medical Science Communication Helper Agent

A Streamlit app for the GOSIM hackathon, focused on medical science communication.

### Pre-hackathon setup
Before the hackathon, the following infrastructure was put in place:
- Streamlit app scaffold.
- Automatic deployment to Hugging Face Spaces via GitHub Actions (push to `main` triggers sync).
- OpenAlex API integration for querying recent biomedical literature by keyword.

### Requirements
- Python 3.13+
- `OPEN_ALEX_KEY` set in your `.env` file (get a free key at [openalex.org/settings/api](https://openalex.org/settings/api))
