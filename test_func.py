import pandas as pd
import numpy as np

def calculate_esg_scores(df, E_weight=0.3, S_weight=0.3, G_weight=0.3):
    # Replace 'N' and 'Y' in boolean columns
    bool_cols = ['ES106', 'ES105', 'ES319', 'SA848']
    df[bool_cols] = df[bool_cols].replace(to_replace=['N', 'Y'], value=[0, 1])

    # Helper function to assign quantile-based scores
    def score_by_quantile(series):
        return pd.qcut(series.rank(method='first'), q=4, labels=[0, 1, 2, 3])

    # Create a copy of the DataFrame to store the scores
    df_scores = df.copy()

    # Apply the quantile-based scoring for each factor column
    exclude = ['Security', 'BI001', 'ES106', 'ES105', 'ES319', 'SA848']
    for col in df_scores.columns:
        if col not in exclude:
            df_scores[col] = score_by_quantile(df_scores[col])
            df_scores[col] = df_scores[col].fillna(0)
            df_scores[col] = df_scores[col].astype(int)

    # Rename columns for scoring
    df_scores.columns = ['Security', 'BI001', 'ES106', 'ES105', 'F0946', 'ES015', 'ES014', 'ES494',
                         'ES020', 'ES021', 'ES043', 'ES047', 'ES063', 'ES096', 'ES319', 'SA848']

    # Calculate the total scores for Environmental, Social, and Governance factors
    E_total = df_scores.loc[:, 'ES106':'ES021'].sum(axis=1)
    S_total = df_scores.loc[:, "ES043":"ES047"].sum(axis=1)
    G_total = df_scores.loc[:, "ES063":"SA848"].sum(axis=1)

    # Calculate the final ESG score based on the E, S, G scores
    Completion_bonus_total = 0  # Adjust this logic if needed
    ESG_Score = E_total * E_weight + S_total * S_weight + G_total * G_weight + Completion_bonus_total

    # Create a DataFrame to store the final scores
    Scores = pd.DataFrame()
    Scores['Securities'] = df_scores['Security']
    Scores['E'] = E_total
    Scores['S'] = S_total
    Scores['G'] = G_total
    Scores['ESG'] = ESG_Score

    return Scores


# Function to loop through all sectors and calculate ESG scores
def calculate_esg_scores_for_all_sectors(dfs, sector_weights):
    combined_scores = pd.DataFrame()

    for i, df_sector in enumerate(dfs):
        sector = df_sector['BI001'].iloc[0]  # Get the sector name
        E_weight, S_weight, G_weight = sector_weights.get(sector, (0.3, 0.3, 0.3))  # Get weights for the sector

        # Calculate ESG scores for the sector
        sector_scores = calculate_esg_scores(df_sector, E_weight=E_weight, S_weight=S_weight, G_weight=G_weight)

        # Add a column for the sector name
        sector_scores['Sector'] = sector

        # Append to the combined DataFrame
        combined_scores = pd.concat([combined_scores, sector_scores], ignore_index=True)

    return combined_scores


# Example usage
# Define sector-specific weights
sector_weights = {
    'Communications': (0.4, 0.3, 0.3),
    'Consumer Discretionary': (0.3, 0.4, 0.3),
    'Consumer Staples': (0.3, 0.3, 0.4),
    # Add weights for other sectors as needed
}

# Call the function to calculate ESG scores for all sectors
combined_esg_scores = calculate_esg_scores_for_all_sectors(dfs, sector_weights)

# View the combined ESG scores
print(combined_esg_scores.head())
