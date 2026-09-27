"""
Admin page — shows unanswered questions and open tickets.
Streamlit auto-discovers this under the sidebar as a second page.
"""
import streamlit as st

from src.handoff import get_all_tickets, get_unanswered_questions
from src.ingest import get_manifest

st.set_page_config(page_title="Admin — Support Bot", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin Dashboard")

col1, col2, col3 = st.columns(3)

manifest = get_manifest()
tickets = get_all_tickets()
unanswered = get_unanswered_questions()

col1.metric("Documents indexed", len(manifest))
col2.metric("Open tickets", len([t for t in tickets if t["status"] == "open"]))
col3.metric("Unanswered questions", len(unanswered))

st.divider()

st.subheader("❓ Unanswered Questions")
if unanswered:
    for entry in reversed(unanswered[-50:]):
        st.markdown(f"- **{entry['question']}**  \n  <sub>session `{entry['session_id'][:8]}` · {entry['timestamp']}</sub>",
                    unsafe_allow_html=True)
else:
    st.caption("No unanswered questions logged yet.")

st.divider()

st.subheader("🎫 Support Tickets")
if tickets:
    for t in reversed(tickets):
        with st.expander(f"Ticket #{t['id']} — {t['reason']} — {t['status']}"):
            st.write(f"**Question:** {t['question']}")
            st.write(f"**Session:** {t['session_id']}")
            st.write(f"**Created:** {t['created_at']}")
else:
    st.caption("No tickets yet.")

st.divider()

st.subheader("📄 Indexed Documents")
if manifest:
    for m in manifest:
        st.markdown(f"- {m['filename']} — indexed {m['ingested_at']}")
else:
    st.caption("No documents indexed yet.")
