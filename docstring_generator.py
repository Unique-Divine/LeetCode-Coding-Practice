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

