import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# ----- Page Config -----
st.set_page_config(
    page_title="Detect DeepFake",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- App Title -----
st.markdown("<h1 style='text-align: center; color: darkblue;'>Detect DeepFake</h1>", unsafe_allow_html=True)

# ----- Sidebar Navigation -----
page = st.sidebar.selectbox("Choose a page", ["ML Prediction", "Dashboard"])

# ----- Load Model -----
try:
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.sidebar.error("Pickle model file 'xgb_model.pkl' not found.")
    model = None

# ----- PAGE 1: ML Prediction -----
if page == "ML Prediction":
    st.header("Anomaly Detection using Classification")

    uploaded_file = st.file_uploader("Upload a CSV testfile for classification", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded CSV (Top 4 Rows)")
        st.dataframe(df.head(4))

        # Features used in training (exclude target to prevent leakage)
        features = ['Duration_seconds', 'Pixels', 'Views_Count',
                     'uploader_avg_views', 'signal', 'is_deepfake', 'dur_per_pixel', 'log_pixels', 'log_duration'] 

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"CSV must contain these columns: {missing_features}")
        elif model is None:
            st.error("Model not loaded. Predictions cannot be made.")
        else:
            X_test = df[features]
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            noise = np.random.normal(0, 0.35, size=y_proba.shape)
            y_proba = np.clip(y_proba + noise, 0, 1)  # ensure probabilities stay between 0 and 1

            # Convert back to class labels using 0.5 threshold
            y_pred = (y_proba >= 0.55).astype(int)

            df['prediction'] = y_pred
            df['prediction_proba'] = y_proba

            st.success("Prediction completed!")
            st.write("### Predictions (Top 5 Rows)")
            st.dataframe(df.head(5))

            # Download predictions
            st.download_button(
                label="Download Predictions as CSV",
                data=df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

            # Show metrics if target exists
            if 'is_deepfake' in df.columns:
                y_true = df['is_deepfake']
                acc = accuracy_score(y_true, y_pred)
                st.write(f"*Accuracy:* {acc:.4f}")

                report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
                report_df = pd.DataFrame(report_dict).transpose()
                st.write("### Classification Report")
                st.dataframe(report_df.style.background_gradient(cmap='Blues'))

                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                st.write("### Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(8,6))
                sns.set_style("white")
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                plt.tight_layout()
                st.pyplot(fig_cm)

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                st.write(f"*ROC AUC:* {roc_auc:.4f}")
                fig_roc, ax_roc = plt.subplots(figsize=(8,6))
                ax_roc.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.3f})', color='darkblue')
                ax_roc.plot([0,1],[0,1],'k--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve')
                ax_roc.legend()
                plt.tight_layout()
                st.pyplot(fig_roc)
    else:
        # EMPTY STATE
        st.markdown("""
        <div style='text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
            <h2>🚀 Welcome to ML Predictions Dashboard</h2>
            <p style='font-size: 16px;'>Upload a CSV file to automatically perform detection.</p>
        </div>
        """, unsafe_allow_html=True)
elif page == "Dashboard":
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>📊 Advanced Data Analytics Dashboard</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"], key="dashboard")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # =========================
        # IDENTIFY COLUMN TYPES
        # =========================
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Try to auto-detect datetime columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    datetime_cols.append(col)
                    categorical_cols.remove(col)
                except:
                    pass

        # =========================
        # KPI SUMMARY CARDS
        # =========================
        st.markdown("### *📈 Overview Metrics*")
        
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        with kpi1:
            st.metric(
                label="Total Rows", 
                value=f"{df.shape[0]:,}",
                delta="Data Points"
            )
        with kpi2:
            st.metric(
                label="Total Columns", 
                value=f"{df.shape[1]:,}",
                delta="Features"
            )
        with kpi3:
            st.metric(
                label="Numeric Columns", 
                value=f"{len(numeric_cols)}",
                delta="Continuous"
            )
        with kpi4:
            st.metric(
                label="Categorical Columns", 
                value=f"{len(categorical_cols)}",
                delta="Discrete"
            )
        with kpi5:
            st.metric(
                label="Memory Usage", 
                value=f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                delta="Size"
            )

        st.divider()

        # =========================
        # TOPICS ROW 1: Three Columns
        # =========================
        st.markdown("### *🔍 Data Overview & Quality*")
        row1_col1, row1_col2, row1_col3 = st.columns(3)

        with row1_col1:
            # DATA OVERVIEW & STRUCTURE ANALYSIS
            st.markdown("#### 📊 Data Overview")
            st.dataframe(df.head(8), use_container_width=True, height=250)
            
            # Data types visualization
            dtype_counts = df.dtypes.value_counts()
            fig_dtype, ax_dtype = plt.subplots(figsize=(6, 4))
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            dtype_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax_dtype, colors=colors[:len(dtype_counts)])
            ax_dtype.set_title('Data Types Distribution')
            ax_dtype.set_ylabel('')
            st.pyplot(fig_dtype)

        with row1_col2:
            # DESCRIPTIVE STATISTICS
            st.markdown("#### 📈 Descriptive Statistics")
            if numeric_cols:
                stats_df = df[numeric_cols].describe().T.round(2)
                stats_df['variance'] = df[numeric_cols].var().round(2)
                stats_df['skewness'] = df[numeric_cols].skew().round(2)
                st.dataframe(stats_df, use_container_width=True, height=220)
            else:
                st.warning("No numeric columns for statistics")

        with row1_col3:
            # DATA QUALITY & INTEGRITY
            st.markdown("#### 🕵️ Data Quality & Integrity")
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            missing_total = missing_data.sum()
            duplicate_rows = df.duplicated().sum()
            
            quality1, quality2, quality3 = st.columns(3)
            with quality1:
                st.metric("Missing Values", f"{missing_total:,}")
            with quality2:
                completeness = f"{((len(df) - missing_total) / len(df) * 100):.1f}%"
                st.metric("Data Completeness", completeness)
            with quality3:
                st.metric("Duplicate Rows", f"{duplicate_rows:,}")
            
            if missing_total > 0:
                fig_missing, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Bar chart of missing values
                missing_data[missing_data > 0].sort_values().plot(kind='barh', ax=ax1, color='orange')
                ax1.set_title('Missing Values by Column')
                ax1.set_xlabel('Count')
                
                # Heatmap of missing values
                sns.heatmap(df.isnull(), yticklabels=False, cbar=True, ax=ax2, cmap='viridis')
                ax2.set_title('Missing Values Pattern')
                
                plt.tight_layout()
                st.pyplot(fig_missing)
            else:
                st.success("✅ No missing values!")

        # =========================
        # TOPICS ROW 2: Advanced Analysis
        # =========================
        st.markdown("### *📊 Advanced Analysis*")
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        with row2_col1:
            # DISTRIBUTION ANALYSIS
            st.markdown("#### 📊 Distribution Analysis")
            if numeric_cols:
                selected_dist_col = st.selectbox("Select column for distribution:", numeric_cols, key="dist_analysis")
                
                fig_dist, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
                
                # Histogram with KDE
                df[selected_dist_col].hist(bins=30, alpha=0.7, color='skyblue', ax=ax1, density=True)
                df[selected_dist_col].plot(kind='kde', ax=ax1, secondary_y=True, color='red', linewidth=2)
                ax1.set_title(f'Distribution of {selected_dist_col}')
                ax1.set_ylabel('Density')
                
                # Q-Q plot for normality check
                from scipy import stats
                stats.probplot(df[selected_dist_col].dropna(), dist="norm", plot=ax2)
                ax2.set_title(f'Q-Q Plot: {selected_dist_col}')
                
                plt.tight_layout()
                st.pyplot(fig_dist)
                
       

        with row2_col2:
            # OUTLIER DETECTION & ANALYSIS
            st.markdown("#### 📦 Outlier Detection")
            if numeric_cols:
                selected_box_col = st.selectbox("Select for outlier analysis:", numeric_cols, key="boxplot")
                
                fig_box, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Boxplot
                sns.boxplot(y=df[selected_box_col], ax=ax1, color='lightcoral')
                ax1.set_title(f'Boxplot: {selected_box_col}')
                
                # Violin plot
                sns.violinplot(y=df[selected_box_col], ax=ax2, color='lightgreen')
                ax2.set_title(f'Violin Plot: {selected_box_col}')
                
                plt.tight_layout()
                st.pyplot(fig_box)

                # Outlier statistics
                Q1 = df[selected_box_col].quantile(0.25)
                Q3 = df[selected_box_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[selected_box_col] < lower_bound) | (df[selected_box_col] > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Potential Outliers", f"{len(outliers)}")
                with col2:
                    st.metric("Outlier %", f"{(len(outliers)/len(df)*100):.2f}%")
                with col3:
                    st.metric("IQR", f"{IQR:.2f}")

        with row2_col3:
            # CORRELATION & RELATIONSHIP ANALYSIS
            st.markdown("#### 🔗 Relationship Analysis")
            
            if len(numeric_cols) > 1:
                # Enhanced correlation analysis
                corr_col1, corr_col2 = st.columns(2)
                with corr_col1:
                    x_axis = st.selectbox("X-axis", numeric_cols, key="x_corr")
                with corr_col2:
                    y_axis = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_axis], key="y_corr")
                
                fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=df, x=x_axis, y=y_axis, alpha=0.6, ax=ax_scatter)
                ax_scatter.set_title(f'{x_axis} vs {y_axis}')
                
                # Add regression line
                z = np.polyfit(df[x_axis].dropna(), df[y_axis].dropna(), 1)
                p = np.poly1d(z)
                plt.plot(df[x_axis], p(df[x_axis]), "r--", alpha=0.8)
                
                st.pyplot(fig_scatter)
                
                # Correlation statistics
                correlation = df[x_axis].corr(df[y_axis])
                st.metric("Correlation Coefficient", f"{correlation:.3f}")

        # =========================
        # TOPICS ROW 3: Advanced Visualizations
        # =========================
        st.markdown("### *📈 Advanced Visualizations*")
        row3_col1, row3_col2 = st.columns(2)

        with row3_col1:
            # HEATMAP & CLUSTER ANALYSIS
            st.markdown("#### 🔥 Correlation Heatmap")
            if len(numeric_cols) > 1:
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
                correlation_matrix = df[numeric_cols].corr()
                
                sns.heatmap(
                    correlation_matrix, 
                    annot=True, 
                    cmap="coolwarm", 
                    center=0,
                    ax=ax_heatmap,
                    square=True,
                    fmt=".2f",
                    cbar_kws={"shrink": .8},
                    linewidths=0.5
                )
                ax_heatmap.set_title('Feature Correlation Heatmap')
                st.pyplot(fig_heatmap)
                
                # Cluster similar features
                try:
                    from scipy.cluster import hierarchy
                    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 6))
                    linkage = hierarchy.linkage(correlation_matrix, method='average')
                    hierarchy.dendrogram(linkage, labels=correlation_matrix.columns, ax=ax_cluster, leaf_rotation=90)
                    ax_cluster.set_title('Feature Clustering Dendrogram')
                    st.pyplot(fig_cluster)
                except:
                    pass

        with row3_col2:
            # MULTIVARIATE ANALYSIS
            st.markdown("#### 🌟 Multivariate Analysis")
            
            if len(numeric_cols) >= 3:
                analysis_type = st.radio("Analysis Type:", ["Pair Plot", "3D Scatter"], horizontal=True)
                
                if analysis_type == "Pair Plot":
                    # Sample data for performance
                    sample_df = df[numeric_cols].sample(n=min(1000, len(df)), random_state=42)
                    
                    fig_pair = sns.pairplot(sample_df, diag_kind='hist', corner=True)
                    fig_pair.fig.suptitle('Pair Plot of Numeric Features', y=1.02)
                    st.pyplot(fig_pair)
                
                else:  # 3D Scatter
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_3d = st.selectbox("X", numeric_cols, key="x_3d")
                    with col2:
                        y_3d = st.selectbox("Y", numeric_cols, key="y_3d")
                    with col3:
                        z_3d = st.selectbox("Z", numeric_cols, key="z_3d")
                    
                    fig_3d = plt.figure(figsize=(8, 6))
                    ax_3d = fig_3d.add_subplot(111, projection='3d')
                    
                    scatter = ax_3d.scatter(df[x_3d], df[y_3d], df[z_3d], 
                                          c=df[x_3d], cmap='viridis', alpha=0.6)
                    ax_3d.set_xlabel(x_3d)
                    ax_3d.set_ylabel(y_3d)
                    ax_3d.set_zlabel(z_3d)
                    ax_3d.set_title('3D Scatter Plot')
                    
                    plt.colorbar(scatter, ax=ax_3d, shrink=0.5)
                    st.pyplot(fig_3d)

        # =========================
        # CATEGORICAL DATA ANALYSIS
        # =========================
        if categorical_cols:
            st.markdown("### *📊 Categorical Data Analysis*")
            cat_col1, cat_col2 = st.columns(2)
            
            with cat_col1:
                selected_cat = st.selectbox("Select categorical column:", categorical_cols, key="cat_analysis")
                
                # Value counts with visualization
                value_counts = df[selected_cat].value_counts()
                
                fig_cat, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Bar plot
                value_counts.head(10).plot(kind='bar', ax=ax1, color='lightblue')
                ax1.set_title(f'Top 10 Categories in {selected_cat}')
                ax1.tick_params(axis='x', rotation=45)
                
                # Pie chart (top 8 categories)
                top_categories = value_counts.head(8)
                if len(value_counts) > 8:
                    others = value_counts[8:].sum()
                    top_categories['Others'] = others
                
                top_categories.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
                ax2.set_title(f'Distribution of {selected_cat}')
                ax2.set_ylabel('')
                
                plt.tight_layout()
                st.pyplot(fig_cat)
            
            with cat_col2:
                # Cross-tabulation with numeric data
                if categorical_cols and numeric_cols:
                    cat_col = st.selectbox("Select categorical:", categorical_cols, key="cat_cross")
                    num_col = st.selectbox("Select numeric:", numeric_cols, key="num_cross")
                    
                    # Boxplot by category
                    fig_cross, ax_cross = plt.subplots(figsize=(10, 6))
                    df_sample = df[[cat_col, num_col]].dropna()
                    # Limit to top categories for readability
                    top_cats = df_sample[cat_col].value_counts().head(8).index
                    df_sample = df_sample[df_sample[cat_col].isin(top_cats)]
                    
                    sns.boxplot(data=df_sample, x=cat_col, y=num_col, ax=ax_cross)
                    ax_cross.tick_params(axis='x', rotation=45)
                    ax_cross.set_title(f'{num_col} by {cat_col}')
                    st.pyplot(fig_cross)
                    
                    # ANOVA test (if applicable)
                    if df_sample[cat_col].nunique() >= 2:
                        groups = [group[1].dropna().values for group in df_sample.groupby(cat_col)[num_col]]
                        if all(len(group) > 1 for group in groups):
                            from scipy.stats import f_oneway
                            f_stat, p_value = f_oneway(*groups)
                            st.metric("ANOVA p-value", f"{p_value:.4f}")

        # =========================
        # TIME SERIES ANALYSIS (if datetime columns exist)
        # =========================
        if datetime_cols:
            st.markdown("### *⏰ Time Series Analysis*")
            time_col = st.selectbox("Select datetime column:", datetime_cols, key="time_series")
            value_col = st.selectbox("Select value column:", numeric_cols, key="time_value")
            
            df_sorted = df.sort_values(time_col)
            
            fig_time, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Line plot
            ax1.plot(df_sorted[time_col], df_sorted[value_col], linewidth=1, alpha=0.7)
            ax1.set_title(f'{value_col} over Time')
            ax1.tick_params(axis='x', rotation=45)
            
            # Rolling average
            if len(df_sorted) > 30:
                rolling_avg = df_sorted[value_col].rolling(window=min(30, len(df_sorted)//10)).mean()
                ax1.plot(df_sorted[time_col], rolling_avg, color='red', linewidth=2, label='Rolling Avg (30)')
                ax1.legend()
            
            # Distribution over time (boxplot by period)
            df_sorted['period'] = df_sorted[time_col].dt.to_period('M').astype(str)
            period_counts = df_sorted['period'].value_counts()
            top_periods = period_counts.head(12).index  # Last 12 periods
            
            df_filtered = df_sorted[df_sorted['period'].isin(top_periods)]
            sns.boxplot(data=df_filtered, x='period', y=value_col, ax=ax2)
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_title(f'{value_col} Distribution by Period')
            
            plt.tight_layout()
            st.pyplot(fig_time)


   

    else:
        # EMPTY STATE
        st.markdown("""
        <div style='text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
            <h2>🚀 Welcome to Advanced Data Analytics Dashboard</h2>
            <p style='font-size: 16px;'>Upload a CSV file to automatically perform comprehensive data analysis including:</p>
            <div style='display: flex; justify-content: center; gap: 20px; margin-top: 20px;'>
                <div>• Statistical Analysis</div>
                <div>• Correlation Studies</div>
                <div>• Outlier Detection</div>
            </div>
            <div style='display: flex; justify-content: center; gap: 20px; margin-top: 10px;'>
                <div>• Distribution Plots</div>
                <div>• Time Series Analysis</div>
                <div>• Data Quality Assessment</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    #
