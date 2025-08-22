import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# ---------------------------
# Helper: Draw network graph
# ---------------------------
def draw_network(rules, selected_metric="confidence", min_val=0.5):
    """
    Creates an interactive network graph from the association rules.
    """
    G = nx.DiGraph()
    filtered_rules = rules[rules[selected_metric] >= min_val]
    
    for _, row in filtered_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        for ant in antecedents:
            for cons in consequents:
                G.add_edge(ant, cons, weight=row[selected_metric], title=f"Lift: {row['lift']:.2f}")

    net = Network(height="550px", width="100%", bgcolor="#FFFFFF", font_color="Black", directed=True, notebook=False)
    net.from_nx(G)
    net.repulsion(node_distance=420, central_gravity=0.33, spring_length=110, spring_strength=0.10, damping=0.95)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as html_file:
            components.html(html_file.read(), height=575)

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Market Basket Analysis", layout="wide")
st.title("🛒 Market Basket Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your transaction data (CSV file)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    basket = (df.groupby(['order_id', 'product'])['product']
              .count().unstack().reset_index().fillna(0)
              .set_index('order_id'))
    
    basket_sets = (basket > 0).astype(int)

    st.sidebar.header("Algorithm Parameters")
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.02, 0.01)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.2, 0.05)
    min_lift = st.sidebar.slider("Minimum Lift", 1.0, 5.0, 1.0, 0.1)

    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]

    # --- Display Results ---
    st.markdown("---")
    if rules.empty:
        st.warning("No association rules found for the selected parameters. Please try lowering the thresholds in the sidebar.")
    else:
        st.subheader("📊 Top Association Rules")
        display_rules = rules.copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda a: ', '.join(list(a)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda a: ', '.join(list(a)))
        
        st.dataframe(display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(display_rules)
        st.download_button(
            label="📥 Download Rules as CSV",
            data=csv,
            file_name='association_rules.csv',
            mime='text/csv',
        )

        st.markdown("---")
        st.subheader("🕸️ Interactive Network Graph of Product Associations")
        
        st.sidebar.header("Graph Parameters")
        metric_choice = st.sidebar.selectbox("Choose metric for edge weight:", ["confidence", "lift", "support"])
        
        min_val_default = float(rules[metric_choice].min())
        max_val_default = float(rules[metric_choice].max())
        
        min_val = st.sidebar.slider(f"Minimum {metric_choice} to display:", 
                                    min_value=min_val_default, 
                                    max_value=max_val_default, 
                                    value=min_val_default, 
                                    step=0.01)
        
        draw_network(rules, selected_metric=metric_choice, min_val=min_val)

else:
    st.info("Awaiting for a CSV file to be uploaded.")
