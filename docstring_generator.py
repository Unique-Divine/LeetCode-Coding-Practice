def one_hot_encode(X, cat_features):
    """[summary]

    Args:
        X ([type]): [description]
        cat_features ([type]): [description]

    Returns:
        [type]: [description]
    """
    X_cat = X[cat_features]
    # Remove the current categorical feature columns
    for cat in cat_features[:]:
        X = X.drop(cat, axis=1)

    # Replace the nonnumerical columns with one-hot encoded ones.
    for name in cat_features[:]:
        hot_one = pd.get_dummies(X_cat[name], prefix=name)
        X = pd.concat([X, hot_one.set_index(X.index)], axis=1)
    return X

def display_tear_sheet(factor_data, factor_name):
    """[summary]

    Args:
        factor_data ([type]): [description]
        factor_name ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Get the 1-day forward returns for the assets and dates in the factor
    returns_df = get_forward_returns(
        factor_data[factor_name+' momentum_factor'],
        [1], US_EQUITIES)

    # Format the factor and returns data so that we can run it through Alphalens.
    al_data = al.utils.get_clean_factor(
        factor_data[factor_name+' momentum_factor'],
        returns_df, quantiles=5, bins=None,)

    return create_full_tear_sheet(al_data)