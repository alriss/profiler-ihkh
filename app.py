import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
from time import gmtime, strftime
# import openpyxl
# import os

# Set the page config to use the light theme
st.set_page_config(page_title="Profiler? I Hardly Know Her!", page_icon="📊",layout="wide", initial_sidebar_state="auto")

# Function to load the dataset
def load_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            try:
                # Read the first few lines to determine the delimiter
                first_lines = uploaded_file.read(1024).decode('utf-8')
                uploaded_file.seek(0)  # Reset file pointer
                delimiter = ',' if first_lines.count(',') > first_lines.count(';') else ';'
                df = pl.read_csv(uploaded_file, separator=delimiter,truncate_ragged_lines=True,infer_schema_length=10000)
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None
        elif uploaded_file.name.endswith('.xlsx'):
            df = pl.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pl.read_csv(uploaded_file, sep='\t')
        else:
            st.error("Unsupported file format.")
            return None
        tab1, tab2 = st.tabs(["Data Profiler", "Data Explorer"])
        df = df.filter(~pl.all_horizontal(pl.all().is_null()))
        st.session_state.dataframe = df
        return df
    return None

# Function to show data profiling
def show_data_profile(df):
    if df is not None:
        st.subheader("Data Profiling")
        
        # Basic statistics with Polars
        st.write("**Number of rows and columns:**", df.shape)
        
        # Create a summary table for all columns
        summary_data = {
            "Column": [],
            "Type": [],
            "Category":[],
            "Unique Values": [],
            "Non-null values": [],
            "Entropy": [],
            "Efficiency": [],
            "Max": [],
            "Min": [],
            "μ": [],
            "σ": [],
            "Median": [],
            "Mode": [],
            "Skewness": []
        }
        
        for col in df.columns:
            summary_data["Column"].append(col)
            summary_data["Type"].append(str(df[col].dtype))

            total = df.shape[0] - sum(df.select(pl.all().is_null())[col])
            summary_data["Non-null values"].append(total)

            unique_values = df[col].n_unique()
            summary_data["Unique Values"].append(unique_values)
            
            # Compute entropy
            counts = df[col].drop_nulls().value_counts()['count']
            entropy_value = -sum((count / total) * math.log2(count / total) for count in counts)
            summary_data["Entropy"].append(round(entropy_value,2))
            
            # Compute Shannon Density (Normalized Entropy)
            if unique_values > 1:
                efficiency = entropy_value / math.log2(unique_values)
            else:
                efficiency = 0
            summary_data["Efficiency"].append(round(efficiency,2))

            # Determine category
            if unique_values == 1:
                category = "Fixed"
            elif df[col].dtype == pl.Float64:
                category = "Informative"
            elif unique_values == len(df):
                category = "ID-like"
            elif (df[col].dtype in [pl.Utf8, pl.Categorical]) and ((unique_values<=20 and round(efficiency,2)>=.9) or round(entropy_value, 2) == round(math.log2(unique_values), 2)):
                category = "Categorical (balanced)"
            elif (df[col].dtype in [pl.Utf8, pl.Categorical]) and ((unique_values<=20 and round(efficiency,2)<.9) or entropy_value < 1):
                category = "Categorical (imbalanced)"
            elif unique_values / total > 0.9:
                category = "Mostly unique"
            else:
                category = "Informative"
            
            summary_data["Category"].append(category)
            
            summary_data["Max"].append(df[col].max())
            summary_data["Min"].append(df[col].min())
            summary_data["μ"].append(df[col].mean())
            summary_data["Median"].append(df[col].median())

            if df[col].dtype in [pl.Utf8, pl.Categorical]:
                summary_data["σ"].append("")
            else:
                summary_data["σ"].append(round(df[col].std(), 2))

            # Compute Mode
            mode_value = df[col].value_counts().row(0)[0]  # Taking the most frequent value
            summary_data["Mode"].append(mode_value)

            # Compute Skewness
            if df[col].dtype not in [pl.Utf8, pl.Categorical]:
                # Remove None values
                valid_values = [x for x in df[col] if x is not None]
                
                if len(valid_values) > 1:  # Ensure at least 2 values for meaningful skewness
                    mean = sum(valid_values) / len(valid_values)
                    std_dev = (sum((x - mean) ** 2 for x in valid_values) / len(valid_values)) ** 0.5
                    
                    if std_dev > 0:
                        skewness = sum(((x - mean) / std_dev) ** 3 for x in valid_values) / len(valid_values)
                    else:
                        skewness = 0  # If standard deviation is 0, skewness is undefined
                    
                    summary_data["Skewness"].append(round(skewness, 2))
                else:
                    summary_data["Skewness"].append("")  # Not enough data for skewness
            else:
                summary_data["Skewness"].append("")

        # Convert summary data to a Pandas DataFrame for display
        summary_df = pd.DataFrame(summary_data)
        # st.table(summary_df)
        st.dataframe(summary_df, height=200)

        
        # Heatmap for null values
        
        # Create heatmap data (convert to 1s and 0s for null presence)
        heatmap_data = df.select(pl.all().is_null()).transpose().to_numpy().astype(int)

        # Compute null value frequency (%) by row
        null_frequencies_by_row = (heatmap_data.sum(axis=0) / df.shape[1]) * 100
        x_vals = list(range(len(null_frequencies_by_row)))

        # Compute null value frequency (%) by column
        null_frequencies_by_column = (df.select(pl.all().is_null()).sum().to_numpy().flatten() / df.shape[0]) * 100
        y_vals = list(range(len(null_frequencies_by_column)))

        row_height=30
        freq_height=90

        # Create subplots with 3 panels: heatmap, horizontal line chart (row nulls), vertical line chart (column nulls)
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            row_heights=[row_height*len(df.columns)/(freq_height+row_height*len(df.columns)), freq_height/(freq_height+row_height*len(df.columns))],  # Make right panel smaller
            column_widths=[0.85, 0.15],  # Make right panel smaller
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )

        # Add heatmap
        heatmap_trace = go.Heatmap(
            z=heatmap_data,
            colorscale=["#01203b", "white"],  # Missing values are white
            showscale=False,
            ygap=2
        )
        fig.add_trace(heatmap_trace, row=1, col=1)

        # Add actual null frequency bars
        bar_trace_col = go.Bar(
            x=x_vals,  # Main data
            y=null_frequencies_by_row,
            marker=dict(color="#01203b"),
            name="Nulls (%) by Row"
        )

        # Add complementary bars (100 - null frequency)
        bar_trace_complement = go.Bar(
            x=x_vals,  # Remaining part
            y=100 - null_frequencies_by_row,
            marker=dict(color="#cfcfcf"),
            name="Non-Null (%)",
            showlegend=False  # Optional: Hide legend for cleaner look
        )

        # Add both traces to the figure
        fig.add_trace(bar_trace_col, row=2, col=1)  # Foreground bar
        fig.add_trace(bar_trace_complement, row=2, col=1)  # Background bar

        # Enable stacking
        fig.update_layout(barmode='stack')

        # Add actual null frequency bars
        bar_trace_col = go.Bar(
            x=null_frequencies_by_column,  # Main data
            y=y_vals,
            orientation='h',  # Horizontal bars
            marker=dict(color="#01203b"),
            name="Nulls (%) by Column"
        )

        # Add complementary bars (100 - null frequency)
        bar_trace_complement = go.Bar(
            x=100 - null_frequencies_by_column,  # Remaining part
            y=y_vals,
            orientation='h',
            marker=dict(color="#cfcfcf"),
            name="Non-Null (%)",
            showlegend=False  # Optional: Hide legend for cleaner look
        )

        # Add both traces to the figure
        fig.add_trace(bar_trace_col, row=1, col=2)  # Foreground bar
        fig.add_trace(bar_trace_complement, row=1, col=2)  # Background bar

        # Enable stacking
        fig.update_layout(barmode='stack')

        # Update layout
        fig.update_layout(
            xaxis2=dict(title="Nulls (%) by Column",showticklabels=True),  # X-axis for row null frequency chart
            # xaxis=dict(showticklabels=False),
            # yaxis=dict(showticklabels=False),
            yaxis=dict(
                title="Columns",
                tickmode="array",
                tickvals=list(range(len(df.columns))),
                ticktext=df.columns
            ),
            yaxis2=dict(autorange="reversed"),
            xaxis3=dict(title="Nulls (%) by Row"),  # X-axis for column null frequency chart
            yaxis3=dict(autorange="reversed",showticklabels=True),  # Hide y-axis ticks for column null frequency chart
            title='Heatmap',
            height=(100+freq_height+row_height*len(df.columns))
        )

        # Show figure
        st.plotly_chart(fig)

        fig = make_subplots(
            rows=round(len(df.columns)-.5)+1, cols=3,
            subplot_titles=df.columns,
            vertical_spacing=0.05
        )

        for i,c in enumerate(df.columns):
            hist = go.Histogram(
                x=df[c], 
                nbinsx= None if df[c].dtype in [pl.Float64, pl.Int64] else df[c].n_unique(), 
                histnorm='probability density',
                marker=dict(color="#01203b")  # Set color for all histograms
            )
            fig.add_trace(hist, row=(i//3)+1, col=(i%3)+1)
        fig.update_layout(
            height=200*round(len(df.columns)-.5)+1,
            showlegend=False,  # Remove legend
            title_text="Histograms of Dataset Columns",
            bargap=0.01
        )
        st.plotly_chart(fig)

# Function to show an interactive plot
# Function to show an interactive plot
def show_interactive_plot(df, uploaded_file):
    if df is not None:
        st.subheader("Interactive Plot")
        
        # Reset session state when a new file is uploaded
        if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != uploaded_file.name:
            st.session_state.uploaded_file = uploaded_file.name
            st.session_state.x_axis = df.columns[0]  # Default to the first column
            st.session_state.y_axis = df.columns[1]  # Default to the second column
        
        # Select columns for the plot
        columns = df.columns
        st.session_state.x_axis = st.selectbox("Select X-axis column", columns, index=columns.index(st.session_state.x_axis))
        st.session_state.y_axis = st.selectbox("Select Y-axis column", columns, index=columns.index(st.session_state.y_axis))
        
        # Interactive plot with Plotly
        plot = px.scatter(df.to_pandas(), x=st.session_state.x_axis, y=st.session_state.y_axis)
        st.plotly_chart(plot)

# Function to export the report
def export_report(df, uploaded_file):
    if df is not None:
        # Export profiling to text file
        with open(f"reports/{uploaded_file.name}_{strftime("%Y%m%d_%H%M%S", gmtime())}_profile.txt", "w") as f:
            f.write("Data Profiling Report\n\n")
            f.write(f"File: {uploaded_file.name}\n")
            f.write(f"Number of rows and columns: {df.shape}\n\n")
            f.write("Columns:\n")
            for c in df.columns:
                f.write(f"- {c}\n")

            f.write("\n")
            for col in df.columns:
                f.write(f"Column:\t\t{col}\n")
                f.write(f"Type:\t\t{df[col].dtype}\n")
                f.write(f"Unique values:\t{df[col].n_unique()}\n")
                f.write(f"Max:\t\t{df[col].max()}\n")
                f.write(f"Min:\t\t{df[col].min()}\n")
                f.write(f"Mean:\t\t{df[col].mean()}\n")
                f.write(f"Median:\t\t{df[col].median()}\n")
                if df[col].dtype in [pl.Utf8, pl.Categorical]:
                    f.write("St deviation:\tNone\n")
                else:
                    f.write(f"St deviation:\t{df[col].std()}\n")
                f.write("---\n")

        st.success(f"The report \"{uploaded_file.name}_{strftime("%Y%m%d_%H%M%S", gmtime())}_profile.txt\"has been saved.")

def uploaded():
    st.session_state.uploader = not(st.session_state.uploader) if 'uploader' in st.session_state else True

# Streamlit interface
def main():

    st.markdown("""
        <style>
        .big-font {
            font-size:40px !important;
            font-weight: bold;
        }
        .stFileUploader > label {
        """ + f'{'display: none;' if 'uploader' in st.session_state and st.session_state.uploader else ''}' + """
        }
        .stFileUploader > section {
        """ + f'{'display: none;' if 'uploader' in st.session_state and st.session_state.uploader else ''}' + """
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Profiler :grey[:small[I Hardly Know Her!]]")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, TXT)",accept_multiple_files=False,on_change=uploaded, type=["csv", "xlsx", "txt"]) 

    # Load dataset
    df = load_file(uploaded_file) 

    if df is not None:
        sql_query = st.text_area('SQL Query',placeholder='SELECT * FROM self WHERE ...',key="sql_query")
        df = df.sql(sql_query) if st.session_state.sql_query.strip() and st.session_state.filter=='filtered' else st.session_state.dataframe
        if "filter" not in st.session_state:
            st.session_state.filter = 0

        if st.session_state.filter == 0:
            if st.button('Apply filters', use_container_width=True) and sql_query.strip():
                st.session_state.filter = 'filtered'
                st.rerun()

        elif st.session_state.filter == 'filtered':
            if st.button('Delete filters', use_container_width=True):
                st.session_state.filter = 0
                st.rerun()

        st.write("Here is a preview of your data:")
        st.dataframe(df.to_pandas(), height=200)

        left, middle, right = st.columns(3)

        # Initialize session state for active button
        if "active_button" not in st.session_state:
            st.session_state.active_button = 0

        # Button clicks update session state
        if left.button("Data Profiling", use_container_width=True):
            st.session_state.active_button = 1
        if middle.button("Interactive Plot", use_container_width=True):
            st.session_state.active_button = 2
        if right.button("Export Report", use_container_width=True):
            st.session_state.active_button = 3

        # Execute the corresponding function based on the active button
        if st.session_state.active_button == 1:
            show_data_profile(df)
        if st.session_state.active_button == 2:
            show_interactive_plot(df, uploaded_file)
        if st.session_state.active_button == 3:
            export_report(df, uploaded_file)

if __name__ == "__main__":
    main()
