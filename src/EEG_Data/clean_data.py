import pandas as pd


def clean_df(df: pd.DataFrame, std_threshold: float = 3.0) -> pd.DataFrame:
    """
    Clean a DataFrame by removing anomalies in each column based on a standard deviation threshold
    and filling in missing values with linear interpolation.

    Parameters:
    - df: pd.DataFrame - The input DataFrame with numeric data
    - std_threshold: float - The number of standard deviations from the mean to identify outliers

    Returns:
    - pd.DataFrame - The cleaned DataFrame
    """
    cleaned_df = df.copy()

    for col in cleaned_df.columns:
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            mean = cleaned_df[col].mean()
            std_dev = cleaned_df[col].std()
            threshold_upper = mean + std_threshold * std_dev
            threshold_lower = mean - std_threshold * std_dev

            # Mask anomalies as NaN
            cleaned_df[col] = cleaned_df[col].where(
                (cleaned_df[col] <= threshold_upper)
                & (cleaned_df[col] >= threshold_lower),
                other=pd.NA,
            )

    # Apply linear interpolation to fill NaN values created by removing anomalies
    cleaned_df = cleaned_df.interpolate(method="linear", limit_direction="both")

    return cleaned_df
