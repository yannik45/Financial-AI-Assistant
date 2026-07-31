import pandas as pd

from financial_ai.ml.category_split import CategoryDataSplits


def split_german_training_data_by_merchant(
    generated_data: pd.DataFrame,
    random_state: int = 42,
) -> CategoryDataSplits:
    """Split German training rows while keeping merchant families together."""
    train_groups = set()
    validation_groups = set()
    test_groups = set()

    categories = sorted(generated_data["target_category"].unique())

    for cat in categories:
        category_mask = generated_data["target_category"].eq(cat)
        category_groups = sorted(
            generated_data.loc[
                category_mask,
                "merchant_group",
            ].unique()
        )

        group_count = len(category_groups)
        if group_count < 3:
            raise ValueError(f"Category {cat!r} requires at least 3 merchant groups")

        shuffled_groups = (
            pd.Series(category_groups).sample(frac=1, random_state=random_state).tolist()
        )

        held_out_count = max(1, group_count // 8)
        train_end = group_count - 2 * held_out_count
        validation_end = group_count - held_out_count

        train_groups.update(shuffled_groups[:train_end])
        validation_groups.update(shuffled_groups[train_end:validation_end])
        test_groups.update(shuffled_groups[validation_end:])

    train_mask = generated_data["merchant_group"].isin(train_groups)
    validation_mask = generated_data["merchant_group"].isin(validation_groups)
    test_mask = generated_data["merchant_group"].isin(test_groups)

    return CategoryDataSplits(
        train=generated_data.loc[train_mask].reset_index(drop=True),
        validation=generated_data.loc[validation_mask].reset_index(drop=True),
        test=generated_data.loc[test_mask].reset_index(drop=True),
    )
