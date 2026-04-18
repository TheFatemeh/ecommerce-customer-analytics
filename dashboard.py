import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Olist E-commerce Analytics",
    page_icon="🛒",
    layout="wide"
)

# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df_orders_full = pd.read_csv('data/processed/df_orders_full.csv')
    customers = pd.read_csv('data/processed/df_orders_customers.csv')
    items = pd.read_csv('data/processed/df_items_products.csv')

    # Parse timestamps
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date',
                'order_estimated_delivery_date']:
        df_orders_full[col] = pd.to_datetime(df_orders_full[col])

    # Derived columns
    df_orders_full['revenue'] = df_orders_full['price'] + df_orders_full['freight_value']
    df_orders_full['delay'] = (
        df_orders_full['order_delivered_customer_date'] - df_orders_full['order_estimated_delivery_date']
    ).dt.days
    df_orders_full['order_month'] = df_orders_full['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()

    # Delivered orders with revenue
    df_rev = df_orders_full[
        (df_orders_full['order_status'] == 'delivered') &
        df_orders_full['revenue'].notnull()
    ].copy()

    # RFM segmentation
    snapshot_date = df_rev['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df_rev.groupby('customer_unique_id').agg(
        recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('revenue', 'sum')
    )
    rfm['R_segment'] = pd.qcut(rfm['recency'], 3, labels=['recent', 'mid', 'old'])
    rfm['F_segment'] = pd.qcut(rfm['frequency'].rank(method='first'), 3, labels=['low', 'mid', 'high'])
    rfm['M_segment'] = pd.qcut(rfm['monetary'], 3, labels=['low', 'mid', 'high'])

    def assign_segment(row):
        if row['F_segment'] == 'high' and row['M_segment'] == 'high':
            return 'High Value'
        elif row['R_segment'] == 'old' and row['F_segment'] == 'low':
            return 'At Risk'
        elif row['F_segment'] == 'low' and row['M_segment'] == 'low':
            return 'Low Value'
        else:
            return 'Mid Value'

    rfm['segment'] = rfm.apply(assign_segment, axis=1)
    df_rev = df_rev.merge(rfm[['segment']], on='customer_unique_id', how='left')

    # Add customer state
    state_map = customers[['order_id', 'customer_state']].drop_duplicates('order_id')
    df_rev = df_rev.merge(state_map, on='order_id', how='left')

    # Category table: item-level price + order-level review score
    orders_cat = items.merge(
        df_rev[['order_id', 'review_score', 'revenue']],
        on='order_id',
        how='left'
    ).rename(columns={'price': 'item_price'})

    return df_rev, rfm, orders_cat


df_rev, rfm, orders_cat = load_data()

# ── Sidebar Navigation ────────────────────────────────────────────────────────

st.sidebar.title("Olist Analytics")
st.sidebar.caption("Brazilian E-commerce | 2016–2018")
page = st.sidebar.radio("", [
    "Overview",
    "Revenue Trends",
    "Customer Segments",
    "Delivery & Reviews",
    "Category Analysis",
])

# ── PAGE: Overview ────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Olist E-commerce Overview")
    st.caption("Olist Brazilian E-commerce Dataset · Sep 2016 – Aug 2018")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${df_rev['revenue'].sum() / 1e6:.2f}M")
    c2.metric("Total Delivered Orders", f"{len(df_rev):,}")
    c3.metric("Avg Order Value", f"${df_rev['revenue'].mean():.2f}")
    c4.metric("Avg Review Score", f"{df_rev['review_score'].mean():.2f} / 5")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        monthly = df_rev.groupby('order_month')['revenue'].sum().reset_index()
        fig = px.area(
            monthly, x='order_month', y='revenue',
            title='Monthly Revenue', labels={'order_month': '', 'revenue': 'Revenue ($)'}
        )
        fig.update_traces(line_color='#1f77b4', fillcolor='rgba(31,119,180,0.15)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_rev = rfm.groupby('segment')['monetary'].sum().reset_index()
        fig = px.pie(
            seg_rev, values='monetary', names='segment',
            title='Revenue Share by Customer Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 3 Business Insights")
    with st.expander("1. Late delivery is the primary driver of negative reviews"):
        st.write(
            "Late orders are only **6.7%** of deliveries but account for **36.6%** of all "
            "1-star reviews — a **5.5× over-representation**. Fixing logistics consistency "
            "is the highest-leverage action to improve satisfaction."
        )
    with st.expander("2. Repeat buyers spend 2× more — but 96.9% of customers never return"):
        st.write(
            "Only **3.1%** of customers make a second purchase, yet they average **$327** "
            "vs **$161** for one-time buyers. The at-risk segment (11.2% of customers) left "
            "with great reviews — they're dormant, not dissatisfied. A re-engagement campaign "
            "could unlock significant value."
        )
    with st.expander("3. Black Friday permanently shifted the platform's growth rate"):
        st.write(
            "Monthly revenue jumped from ~$400K to a sustained **~$1M+/month plateau** "
            "after November 2017. New customer acquisition also rose from ~700 to ~6,000–7,000 "
            "per month and stayed there — the event durably changed the platform's trajectory."
        )

# ── PAGE: Revenue Trends ──────────────────────────────────────────────────────

elif page == "Revenue Trends":
    st.title("Revenue Trends")

    # Monthly bar chart with Black Friday annotation
    monthly = df_rev.groupby('order_month')['revenue'].sum().reset_index()
    fig = px.bar(
        monthly, x='order_month', y='revenue',
        title='Monthly Revenue Over Time',
        labels={'order_month': 'Month', 'revenue': 'Revenue ($)'}
    )
    bf_date = pd.Timestamp('2017-11-01')
    fig.add_vline(
        x=bf_date.timestamp() * 1000,
        line_dash='dash', line_color='red',
        annotation_text='Black Friday', annotation_position='top right'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Daily revenue + 30-day rolling average
    daily = (
        df_rev.groupby(df_rev['order_purchase_timestamp'].dt.date)['revenue']
        .sum().reset_index()
    )
    daily.columns = ['date', 'revenue']
    daily['date'] = pd.to_datetime(daily['date'])
    daily['rolling_30'] = daily['revenue'].rolling(30).mean()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=daily['date'], y=daily['revenue'],
        name='Daily Revenue', opacity=0.35,
        line=dict(color='lightblue', width=1)
    ))
    fig2.add_trace(go.Scatter(
        x=daily['date'], y=daily['rolling_30'],
        name='30-Day Rolling Avg', line=dict(color='#1f77b4', width=2.5)
    ))
    fig2.update_layout(
        title='Daily Revenue with 30-Day Rolling Average',
        xaxis_title='Date', yaxis_title='Revenue ($)',
        legend=dict(orientation='h', y=1.02)
    )
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Black Friday Revenue (Nov 24, 2017)", "$177,178", "+2.5× next day")
    c2.metric("Peak Month (Nov 2017)", "$1.16M")
    c3.metric("Revenue Growth (Jan 2017 → mid-2018)", "~8×")

# ── PAGE: Customer Segments ───────────────────────────────────────────────────

elif page == "Customer Segments":
    st.title("Customer Segmentation (RFM)")
    st.caption("Segments based on Recency, Frequency, and Monetary value of delivered orders.")

    col1, col2 = st.columns(2)

    with col1:
        seg_counts = rfm['segment'].value_counts().reset_index()
        seg_counts.columns = ['segment', 'count']
        fig = px.pie(
            seg_counts, values='count', names='segment',
            title='Customer Count by Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_rev = rfm.groupby('segment')['monetary'].sum().reset_index()
        fig = px.pie(
            seg_rev, values='monetary', names='segment',
            title='Revenue Share by Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # Summary table
    seg_table = rfm.groupby('segment').agg(
        Customers=('monetary', 'count'),
        Avg_Spend=('monetary', 'mean'),
        Total_Revenue=('monetary', 'sum'),
        Avg_Recency_Days=('recency', 'mean'),
        Avg_Frequency=('frequency', 'mean'),
    ).round(2).reset_index()
    seg_table.columns = ['Segment', 'Customers', 'Avg Spend ($)', 'Total Revenue ($)', 'Avg Recency (days)', 'Avg Frequency']
    seg_table['Avg Spend ($)'] = seg_table['Avg Spend ($)'].map('${:,.2f}'.format)
    seg_table['Total Revenue ($)'] = seg_table['Total Revenue ($)'].map('${:,.0f}'.format)
    seg_table['Avg Recency (days)'] = seg_table['Avg Recency (days)'].map('{:.0f}'.format)
    st.dataframe(seg_table, use_container_width=True, hide_index=True)

    st.info(
        "**At-Risk customers** (11.2% of base) have the highest avg review score (4.13) and "
        "received deliveries 13 days early on average — they left satisfied but haven't returned. "
        "A re-engagement campaign is worth testing on this group."
    )

    # Repeat vs one-time
    st.subheader("Repeat vs One-Time Buyers")
    customer_orders = (
        df_rev.groupby('customer_unique_id')
        .agg(num_orders=('order_id', 'count'), total_spend=('revenue', 'sum'))
        .reset_index()
    )
    repeat = customer_orders[customer_orders['num_orders'] > 1]
    one_time = customer_orders[customer_orders['num_orders'] == 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("One-Time Customers", f"{len(one_time):,}", f"{len(one_time)/len(customer_orders)*100:.1f}%")
    c2.metric("Repeat Customers", f"{len(repeat):,}", f"{len(repeat)/len(customer_orders)*100:.1f}%")
    c3.metric("Avg Spend: Repeat vs One-Time",
              f"${repeat['total_spend'].mean():.0f}",
              f"vs ${one_time['total_spend'].mean():.0f} one-time")

# ── PAGE: Delivery & Reviews ──────────────────────────────────────────────────

elif page == "Delivery & Reviews":
    st.title("Delivery Performance & Customer Satisfaction")

    delivered = df_rev[df_rev['delay'].notnull() & df_rev['review_score'].notnull()].copy()
    late = delivered[delivered['delay'] > 0]
    on_time = delivered[delivered['delay'] <= 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On-Time Rate", f"{len(on_time) / len(delivered) * 100:.1f}%")
    c2.metric("Avg Review (On-Time)", f"{on_time['review_score'].mean():.2f}")
    c3.metric("Avg Review (Late)", f"{late['review_score'].mean():.2f}")
    c4.metric("Late → 1-Star Over-representation", "5.5×")

    st.markdown("---")

    # Review score by delay bucket
    bins = [-999, -7, -1, 0, 3, 7, 14, 999]
    labels = ['Early 7+ d', 'Early 2–7 d', 'Early 0–1 d',
              'Late 1–3 d', 'Late 4–7 d', 'Late 8–14 d', 'Late 14+ d']
    delivered['delay_bucket'] = pd.cut(delivered['delay'], bins=bins, labels=labels)
    bucket_scores = delivered.groupby('delay_bucket', observed=True)['review_score'].mean().reset_index()

    colors = ['#2196F3', '#42A5F5', '#90CAF9', '#FFB300', '#FB8C00', '#E53935', '#B71C1C']
    fig = px.bar(
        bucket_scores, x='delay_bucket', y='review_score',
        title='Avg Review Score by Delivery Timing',
        labels={'delay_bucket': 'Delivery Timing', 'review_score': 'Avg Review Score'},
        color='delay_bucket', color_discrete_sequence=colors
    )
    fig.update_layout(showlegend=False, yaxis_range=[1, 5.5])
    fig.add_hline(y=4.16, line_dash='dot', line_color='gray', annotation_text='Platform avg (4.16)')
    st.plotly_chart(fig, use_container_width=True)

    # Delay by state
    if 'customer_state' in delivered.columns:
        state_delay = (
            delivered.groupby('customer_state')['delay']
            .mean().sort_values(ascending=False).head(15).reset_index()
        )
        state_delay['color'] = state_delay['delay'].apply(lambda x: 'Late (risk)' if x > 0 else 'Early (safe)')
        fig2 = px.bar(
            state_delay, x='delay', y='customer_state', orientation='h',
            color='color',
            color_discrete_map={'Late (risk)': '#E53935', 'Early (safe)': '#2196F3'},
            title='Avg Delivery Delay by State — Top 15 (negative = early)',
            labels={'delay': 'Days vs Estimated Date', 'customer_state': 'State'}
        )
        fig2.add_vline(x=0, line_color='black', line_width=1)
        st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "States AL, MA, and SE have the thinnest early-delivery buffer (~9 days). "
        "Any logistics disruption there flips directly to late orders and 1-star reviews."
    )

# ── PAGE: Category Analysis ───────────────────────────────────────────────────

elif page == "Category Analysis":
    st.title("Category Performance")

    # Revenue vs review score (top 10)
    cat_perf = (
        orders_cat.groupby('product_category_name')
        .agg(revenue=('revenue', 'sum'), avg_review=('review_score', 'mean'))
        .dropna()
        .sort_values('revenue', ascending=False)
        .head(10)
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_perf['product_category_name'], y=cat_perf['revenue'],
        name='Revenue ($)', marker_color='steelblue', yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=cat_perf['product_category_name'], y=cat_perf['avg_review'],
        name='Avg Review Score', mode='lines+markers',
        line=dict(color='crimson', width=2), marker=dict(size=8), yaxis='y2'
    ))
    fig.update_layout(
        title='Top 10 Categories: Revenue vs Avg Review Score',
        yaxis=dict(title='Revenue ($)'),
        yaxis2=dict(title='Avg Review Score', overlaying='y', side='right', range=[1, 5.5]),
        xaxis_tickangle=-40,
        legend=dict(orientation='h', y=1.1)
    )
    fig.add_hline(
        y=4.16, line_dash='dot', line_color='gray',
        annotation_text='Platform avg review', yref='y2'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**cama_mesa_banho** and **informatica_acessorios** are high-revenue but score "
        "below the platform average — highest priority for listing quality improvements."
    )

    # 1-star rate by category
    cat_1star = (
        orders_cat[orders_cat['review_score'].notnull()]
        .groupby('product_category_name')
        .agg(
            one_star_rate=('review_score', lambda x: (x == 1).mean()),
            count=('review_score', 'count')
        )
        .query('count >= 100')
        .sort_values('one_star_rate', ascending=False)
        .head(10)
        .reset_index()
    )
    cat_1star['1-Star Rate (%)'] = (cat_1star['one_star_rate'] * 100).round(1)

    fig2 = px.bar(
        cat_1star, x='product_category_name', y='1-Star Rate (%)',
        title='Top 10 Categories by 1-Star Review Rate (min 100 orders)',
        labels={'product_category_name': 'Category'},
        color_discrete_sequence=['#E53935']
    )
    fig2.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "**fashion_roupa_masculina** (22.6%) and **moveis_escritorio** (20.4%) have the "
        "highest 1-star rates. These categories need better listing quality: size guides, "
        "photos, and compatibility notes — not faster shipping."
    )
