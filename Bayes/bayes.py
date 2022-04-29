def create_table(df, label_column):
    table = {}

    # determine values for the label
    value_counts = df[label_column].value_counts().sort_index()
    table["class_names"] = value_counts.index.to_numpy()
    table["class_counts"] = value_counts.values

    # determine probabilities for the features
    for feature in df.drop(label_column, axis=1).columns:
        table[feature] = {}

        # determine counts
        counts = df.groupby(label_column)[feature].value_counts()
        df_counts = counts.unstack(label_column)

        # add one count to avoid "problem of rare values"
        if df_counts.isna().any(axis=None):
            df_counts.fillna(value=0, inplace=True)
            df_counts += 1

        # calculate probabilities
        df_probabilities = df_counts / df_counts.sum()
        for value in df_probabilities.index:
            probabilities = df_probabilities.loc[value].to_numpy()
            table[feature][value] = probabilities

    return table

def predict_example(row, lookup_table):
    global disease_probability
    global not_disease_probability
    class_estimates = lookup_table["class_counts"]
    for feature in row.index:

        try:
            value = row[feature]
            probabilities = lookup_table[feature][value]
            # print(class_estimates)

            class_estimates = class_estimates * probabilities
            # print(class_estimates)
            # print(probabilities)

        # skip in case "value" only occurs in test set but not in train set
        # (i.e. "value" is not in "lookup_table")
        except KeyError:
            continue

    index_max_class = class_estimates.argmax()
    if index_max_class == 1:
        not_disease_probability = class_estimates.min() / (class_estimates.min() + class_estimates.max())
        disease_probability = class_estimates.max() / (class_estimates.min() + class_estimates.max())

    if index_max_class == 0:
        disease_probability = class_estimates.min() / (class_estimates.min() + class_estimates.max())
        not_disease_probability = class_estimates.max() / (class_estimates.min() + class_estimates.max())

    prediction = lookup_table["class_names"][index_max_class]

    return prediction, disease_probability, not_disease_probability
