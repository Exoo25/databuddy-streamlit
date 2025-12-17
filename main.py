import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Databuddy Editor")

# ---------------- Helpers ----------------
def clean_df(df):
    """Reset index + flatten column names."""
    df = df.copy().reset_index(drop=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]
    else:
        df.columns = df.columns.astype(str).str.strip()
    return df

def safe_pivot(df, rows, cols, values, aggfunc="sum"):
    """Safe pivot wrapper with cleaning + error handling."""
    df = clean_df(df)
    try:
        pivot = pd.pivot_table(
            df,
            index=rows if rows else None,
            columns=cols if cols else None,
            values=values,
            aggfunc=aggfunc,
        )
        return pivot
    except Exception as e:
        return f"Pivot error: {e}"

def make_chart(pivot, rows, cols, values, chart_type):
    """Generate charts from pivot table using Plotly Express."""
    pivot_reset = pivot.reset_index()
    
    # Flatten columns if multi-index
    if isinstance(pivot_reset.columns, pd.MultiIndex):
        pivot_reset.columns = ["_".join([str(c) for c in col if c]) for col in pivot_reset.columns.values]
    
    # Find correct y-column
    y_col = values
    if cols:  # If pivot has columns, find flattened column with values
        y_col = [c for c in pivot_reset.columns if values in c][0]
    
    # Ensure numeric
    pivot_reset[y_col] = pd.to_numeric(pivot_reset[y_col], errors="coerce").fillna(0)

    if chart_type == "Bar":
        return px.bar(pivot_reset, x=rows, y=y_col,
                      color=cols[0] if cols else None, barmode="group")
    elif chart_type == "Line":
        return px.line(pivot_reset, x=rows, y=y_col,
                       color=cols[0] if cols else None)
    elif chart_type == "Area":
        return px.area(pivot_reset, x=rows, y=y_col,
                       color=cols[0] if cols else None)
    elif chart_type == "Pie":
        return px.pie(pivot_reset, names=rows[0], values=y_col)
    return None


# ---------------- Mode Selection ----------------
mode = st.radio("Choose Mode:", ["Editor (manual grid)", "Upload File"])

# ---------------- Editor Mode ----------------
if mode == "Editor (manual grid)":
    import streamlit as st
    import pandas as pd

    # --- Initialize df in session state ---
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame({
            "Column1": [""] * 5,
            "Column2": [""] * 5,
            "Column3": [""] * 5
        })

    # --- Layout: Two columns (left = editor, right = relative sidebar) ---
    left, right = st.columns([3, 1])

    with left:
        st.subheader("📊 Editable Data")
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            key="main_editor"
        )
        st.session_state.df = edited_df

    with right:
        st.subheader("⚙️ Controls")

        # Add a scrollable, small-height container
        with st.container():
            st.markdown(
                """
                <style>
                div[data-testid="stVerticalBlock"] > div:nth-child(2) {
                    max-height: 300px;
                    overflow-y: auto;
                    padding-right: 5px;
                    border: 1px solid #444;
                    border-radius: 8px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # --- Rename Columns ---
            st.markdown("**Rename Columns**")
            new_cols = {}
            for col in st.session_state.df.columns:
                new_name = st.text_input(f"{col}", col, key=f"rename_col_{col}")
                new_cols[col] = new_name
            st.session_state.df.rename(columns=new_cols, inplace=True)

            # --- Add New Column ---
            st.markdown("**Add Column**")
            new_col = st.text_input("New column name", key="new_col_input")
            if st.button("Add Column", key="add_column_btn"):
                if new_col:
                    if new_col in st.session_state.df.columns:
                        st.warning(f"Column '{new_col}' already exists!")
                    else:
                        st.session_state.df[new_col] = ""  # add empty col
                        st.success(f"Column '{new_col}' added!")
                else:
                    st.warning("Please enter a column name!")



    # ---------------- Functions ----------------
    st.subheader("Functions")
    func = st.selectbox("Choose function", ["Sum", "Average", "Min", "Max", "Count"])
    col = st.selectbox("Choose column", edited_df.columns)

    if st.button("Apply"):
        if func == "Sum":
            result = int(edited_df[col].sum())
        elif func == "Average":
            result = edited_df[col].mean()
        elif func == "Min":
            result = edited_df[col].min()
        elif func == "Max":
            result = edited_df[col].max()
        elif func == "Count":
            result = edited_df[col].count()
        st.success(f"{func} of {col} = {result}")

    # ---------------- Pivot Table ----------------
    st.subheader("Pivot Analyzer (Editor Data)")
    import plotly.graph_objects as go

    
    # Step 1: Select filter columns
    filter_columns = st.multiselect("Select column(s) to filter", edited_df.columns, key="filter_columns")

    # Step 2: Apply filters
    filtered_df = edited_df.copy()
    for col_filter in filter_columns:
        unique_vals = edited_df[col_filter].dropna().unique()
        selected_vals = st.multiselect(f"Filter {col_filter}", unique_vals, default=unique_vals, key=f"filter_{col_filter}")
        filtered_df = filtered_df[filtered_df[col_filter].isin(selected_vals)]

    # Step 3: Pivot table selections
    rows = st.multiselect("Rows", filtered_df.columns, key="editor_rows")
    cols = st.multiselect("Columns", filtered_df.columns, key="editor_cols")
    values = st.selectbox("Values", filtered_df.columns, key="editor_values")
    aggfunc = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"], key="editor_agg")

    # Step 4: Generate pivot
    if rows and values:
        pivot = safe_pivot(filtered_df, rows, cols, values, aggfunc)
        if isinstance(pivot, str):
            st.error(pivot)
        else:
            st.write("Pivot Table:", pivot)

            # ---------------- Chart ----------------
            st.subheader("📈 Make a Chart")
            chart_type = st.selectbox("Choose Chart Type", ["Bar", "Line", "Area", "Pie"], key="chart_editor")
            if st.button("Generate Chart (Editor)"):
                fig = make_chart(pivot, rows, cols, values, chart_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

# ---------------- Upload File Mode ----------------
else:
    file = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])
    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df = clean_df(df)

        st.write("📑 Data Preview", df.head())

        # Step 1: Select filter columns
        filter_columns = st.multiselect("Select column(s) to filter", df.columns, key="file_filter_columns")

        # Step 2: Apply filters
        filtered_df = df.copy()
        for col_filter in filter_columns:
            unique_vals = df[col_filter].dropna().unique()
            selected_vals = st.multiselect(f"Filter {col_filter}", unique_vals, default=unique_vals, key=f"file_filter_{col_filter}")
            filtered_df = filtered_df[filtered_df[col_filter].isin(selected_vals)]

        # Step 3: Pivot selections
        rows = st.multiselect("Rows", filtered_df.columns, key="file_rows")
        cols = st.multiselect("Columns", filtered_df.columns, key="file_cols")
        values = st.selectbox("Values", filtered_df.columns, key="file_values")
        aggfunc = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"], key="file_agg")

        # Step 4: Generate pivot
        if rows and values:
            pivot = safe_pivot(filtered_df, rows, cols, values, aggfunc)
            if isinstance(pivot, str):
                st.error(pivot)
            else:
                st.write("Pivot Table:", pivot)

                st.download_button("📥 Download CSV", pivot.to_csv().encode("utf-8"), "pivot.csv")

                st.subheader("📈 Make a Chart")
                chart_type = st.selectbox("Choose Chart Type", ["Bar", "Line", "Area", "Pie"], key="chart_file")
                if st.button("Generate Chart (File)"):
                    fig = make_chart(pivot, rows, cols, values, chart_type)
st.subheader("DataBuddy - Explain Column")

                        # Function to explain a column
def explain_column(df, column_name):
                            col = df[column_name].replace(r'^\s*$', pd.NA, regex=True)  # Treat empty strings as missing
                            total = len(col)
                            missing = col.isna().sum()
                            dtype = col.dtype

                            explanation = []
                            explanation.append(f"Column '{column_name}' contains {total} rows.")
                            explanation.append(f"{missing} values are missing.")
                            
                            def format_row_list(rows):
                                if len(rows) == 0:
                                    return ""
                                elif len(rows) == 1:
                                    return str(rows[0])
                                else:
                                    # All except last
                                    all_but_last = ", ".join(str(r) for r in rows[:-1])
                                    # Add last with &
                                    return f"{all_but_last} & {rows[-1]}"

                            if missing > 0:
                                missing_indices = col[col.isna()].index.tolist()
                                missing_rows_csv = [i + 2 for i in missing_indices]  # +2 for CSV row numbers
                                formatted_rows = format_row_list(missing_rows_csv)
                                explanation.append(f"Missing value(s) found at line(s) {formatted_rows}")

                            if pd.api.types.is_numeric_dtype(col):
                                min_val = col.min()
                                max_val = col.max()
                                mean_val = col.mean()
                                explanation.append("This is a numeric column.")
                                explanation.append(f"Values range from {min_val} to {max_val}, average {mean_val:.2f}.")
                            elif pd.api.types.is_string_dtype(col):
                                unique = col.nunique(dropna=True)
                                most_common = col.value_counts().idxmax()
                                explanation.append("This is a text column.")
                                explanation.append(f"{unique} unique values. Most common: '{most_common}'.")
                            else:
                                explanation.append(f"Column type: {dtype}")

                            return "\n".join(explanation)
                            
uploaded_file = st.file_uploader(
                                    "Upload your CSV file",
                                    type=["csv"],
                                    key="explain_column_uploader"
                                )
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully!")

    selected_col = st.selectbox(
        "Select a column to explain:",
        df.columns,
        key="explain_column_selectbox"
    )

    if st.button("Explain Column", key="explain_column_button"):
        explanation = explain_column(df, selected_col)
        st.text_area(
            "Column Explanation:",
            explanation,
            height=250,
            key="explain_column_output"
        )
else:
    st.info("Please upload a CSV file to get started.")
