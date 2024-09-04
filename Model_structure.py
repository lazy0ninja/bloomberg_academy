#%%
import pandas as pd
import numpy as np
#%%
df = pd.read_csv("Combined_Sector_DataFrame_with_Sector_Column.csv")
df.drop('DX113', inplace=True, axis=1)
#%%
df.head()
#%%
df = df.iloc[:, :18]
#%%
df.DX113.unique()
#%%
df = df[df.DX113 =='JP']
#%%
df.shape
#%%
df.BI001.unique()
print(len(df.BI001.unique()))
#%%
df.BI001.unique()
#%%
dfs = []

sectors = sorted(df.BI001.unique())

for i in sectors:
    temp = df[df.BI001 == i]
    dfs.append(temp)
#%%
print(dfs[0].columns)
dfs[0].head()
#%% md
# # Model Structure
# 
# Get score -> get weights -> weight the scores -> get E, S, G -> get ESG score
# 
# first i will just try to get the score for communications sector
#%%
communications = dfs[0]
#%%
sectors
#%% md
# ### Get Scores
#%%
# scores are determined based on the quantiles within each factor against the sector

def score_by_quantile(series):
    return pd.qcut(series, q=4, labels=[0, 1, 2, 3])

# Create a copy of the DataFrame to store the scores
df_scores = df.copy()
#%%
# Apply the scoring for each factor column
for col in df_scores.columns:
    # if columns are score related
    if col != 'Securities' and col != 'BI001':
        df_scores[col] = score_by_quantile(df[col])
        df_scores[col] = df_scores[col].astype(int)
#%%
df_scores
#%% md
# ### Get Weights
#%%
# assume that feature selections and eng is done
#%%
# weights 

E_weight = 0.3
S_weight = 0.3
G_weight = 0.3
completion_bonus_weight = 0.1


# factor weights **this differs by sector**
factors_name = df_scores.iloc[:, 2:].columns
#%%
print(factors_name)
len(factors_name)
#%%
factors_weights = np.random.rand(len(factors_name))  # This is what we need to determine
weights_dict = dict(zip(factors_name, factors_weights))

weights  = [weights_dict.get(key) for key in factors_name]
#%% md
# ### Weights Calculation
#%%
for factor in factors_name:
    df_scores[factor] = df_scores[factor] * weights_dict.get(factor)
#%% md
# ### Score Calculation
#%%

# Environmentals
E_total = df_scores.loc[:, 'ES106':'ES021'].sum(axis=1)
S_total = df_scores.loc[:,"ES043":"ES047"].sum(axis=1)
G_total = df_scores.loc[:,"ES063":"SA848"].sum(axis=1)


# rewards for data completeness based on the number of non-null values
Completion_bonus_total = 0

ESG_Score = E_total*E_weight + S_total*S_weight + G_total*G_weight

# create dataframe to store the score - securities, E, S, G, ESG
Scores = pd.DataFrame()
Scores['Securities'] = df_scores['Securities']
Scores['E'] = E_total
Scores['S'] = S_total
Scores['G'] = G_total
Scores['ESG'] = ESG_Score
# also add sector name
#%%
Scores
#%% md
# **possible problem**
#  
# if we change the weights for each sector, score are only comparable within the sector
# so you cannot really get cross-sector insights. This might be critical for sellside equity analyst
# 
# However...
# if we use same weights across sectors, then we can compare the scores but sectors that are "easy ESG" will have higher scores.
#%% md
# ### Run loop to get scores for all sectors
#%%
df = pd.read_csv("Combined_Sector_DataFrame_with_Sector_Column.csv")
df = df.iloc[:, :18] # this is to filter the cols we need (remove traditional FA cols)

# df = df[df.DX113 =='JP']
# df.drop('DX113', inplace=True, axis=1)
df.head()
#%%

dfs = []

sectors = sorted(df.BI001.unique())

for i in sectors:
    temp = df[df.BI001 == i]
    dfs.append(temp)
#%%

# Initialize an empty DataFrame to store the combined sector scores
combined_scores = pd.DataFrame()
#%%
def calculate_esg_scores(df, factors_name, weights_dict, E_weight=0.3, S_weight=0.3, G_weight=0.3, completion_bonus_weight=0.1):
    """
    Calculate the ESG scores for a given DataFrame using predefined factor weights.
    
    Parameters:
    df: DataFrame
        DataFrame containing securities and ESG factors.
    factors_name: list
        List of ESG factor column names.
    weights_dict: dict
        Dictionary mapping ESG factors to their respective weights.
    E_weight: float, optional
        Weight for Environmental score (default is 0.3).
    S_weight: float, optional
        Weight for Social score (default is 0.3).
    G_weight: float, optional
        Weight for Governance score (default is 0.3).
    completion_bonus_weight: float, optional
        Weight for completion bonus score (default is 0.1).
    
    Returns:
    Scores: DataFrame
        DataFrame containing calculated E, S, G, and ESG scores for each security.
    """

    # Helper function to assign quantile-based scores
    def score_by_quantile(series):
        return pd.qcut(series, q=4, labels=[0, 1, 2, 3])

    # Create a copy of the DataFrame to store the scores
    df_scores = df.copy()

    # Apply the quantile-based scoring for each factor column
    for col in factors_name:
        if col != 'Securities' and col != 'BI001':
            df_scores[col] = score_by_quantile(df[col])
            df_scores[col] = df_scores[col].astype(int)

    # Multiply each factor's score by its corresponding weight
    for factor in factors_name:
        df_scores[factor] = df_scores[factor] * weights_dict.get(factor)

    # Calculate the total scores for Environmental, Social, and Governance factors
    E_total = df_scores.loc[:, 'ES106':'ES021'].sum(axis=1)
    S_total = df_scores.loc[:, "ES043":"ES047"].sum(axis=1)
    G_total = df_scores.loc[:, "ES063":"SA848"].sum(axis=1)

    # Calculate rewards for data completeness (Completion bonus)
    Completion_bonus_total = df_scores[factors_name].notna().sum(axis=1) * completion_bonus_weight

    # Calculate the final ESG score based on the E, S, G scores and completion bonus
    ESG_Score = E_total * E_weight + S_total * S_weight + G_total * G_weight + Completion_bonus_total

    # Create a DataFrame to store the final scores
    Scores = pd.DataFrame()
    Scores['Securities'] = df_scores['Securities']
    Scores['E'] = E_total
    Scores['S'] = S_total
    Scores['G'] = G_total
    Scores['ESG'] = ESG_Score

    return Scores
#%%
factors_name = df_scores.iloc[:, 2:].columns
factors_weights = np.random.rand(len(factors_name))  # This is what we need to determine
weights_dict = dict(zip(factors_name, factors_weights))
#%%
# Loop through each sector and its corresponding DataFrame
for sector, df_sector in zip(sectors, dfs):
    # Calculate ESG scores for the current sector
    sector_scores = calculate_esg_scores(df_sector, factors_name, weights_dict)

    # Add a column for the sector name
    sector_scores['Sector'] = sector

    # Append the current sector's scores to the combined DataFrame
    combined_scores = pd.concat([combined_scores, sector_scores], ignore_index=True)
#%%
combined_scores