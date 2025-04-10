import pandas as pd
import pytest
from faker import Faker

from eda.data_generation import CustomProvider  # type: ignore

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 15)
import random

random.seed(1337)


@pytest.fixture
def custom_provider():
    merchant_weights = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05]
    customer_weights = [0.01] * 100
    card_fraud_rate = 0.01
    fake = Faker()
    return CustomProvider(
        fake,
        merchant_weights,
        customer_weights,
        card_fraud_rate,
    )


def test_transaction_type(custom_provider):
    result = custom_provider.transaction_type()
    assert result in ["CP", "CNP", "contactless", "tokenized", "ATM", "recurring"]


def test_merchant_name(custom_provider):
    result = custom_provider.merchant_name()
    assert result in [
        "Amazon",
        "Walmart",
        "Target",
        "Best Buy",
        "Costco",
        "Local Store",
    ]


def test_customer_id(custom_provider):
    result = custom_provider.customer_id()
    assert result.startswith("customer_")
    assert result.split("_")[1].isdigit()


def test_location(custom_provider):
    result = custom_provider.location()
    assert result in ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]


def test_dollar_amount(custom_provider):
    result = custom_provider.dollar_amount()
    assert 1.0 <= result <= 1000.0
    assert isinstance(result, float)


def test_transaction_timestamp(custom_provider):
    result = custom_provider.transaction_timestamp()
    assert isinstance(result, str)
    datetime_format = "%Y-%m-%d %H:%M:%S"
    try:
        pd.to_datetime(result, format=datetime_format)
    except ValueError:
        pytest.fail(f"Timestamp {result} does not match format {datetime_format}")


def test_card_id(custom_provider):
    result = custom_provider.card_id()
    assert isinstance(result, str)
    assert len(result) == 36  # UUID length


def test_merchant_id(custom_provider):
    result = custom_provider.merchant_id()
    assert isinstance(result, str)
    assert len(result) == 36  # UUID length


def test_transaction_id(custom_provider):
    result = custom_provider.transaction_id()
    assert isinstance(result, str)
    assert len(result) == 36  # UUID length


def test_generate_decision(custom_provider):
    result = custom_provider.generate_decision()
    assert result in [0, 1]


def test_generate_transaction(custom_provider):
    result = custom_provider.generate_transaction()
    expected_keys = [
        "transaction_id",
        "card_id",
        "merchant_id",
        "merchant_name",
        "customer_id",
        "location",
        "transaction_type",
        "dollar_amount",
        "transaction_timestamp",
        "decision",
    ]
    assert isinstance(result, dict)
    assert set(result.keys()) == set(expected_keys)


def test_add_fraud_label_one_transaction_per_card(custom_provider):
    data = pd.DataFrame(
        {
            "card_id": [custom_provider.card_id() for _ in range(10)],
            "transaction_id": [custom_provider.transaction_id() for _ in range(10)],
            "transaction_timestamp": [
                custom_provider.transaction_timestamp() for _ in range(10)
            ],
        }
    )
    label_name = "fraud_label"

    result = custom_provider.add_fraud_label(data, label_name=label_name)

    assert label_name in result.columns
    assert set(result[label_name].unique()).issubset({0, 1})
    assert abs(10 * custom_provider.card_fraud_rate - result[label_name].sum()) <= 2


def test_add_fraud_label_two_transactions_per_card(custom_provider):
    num_cards = 10
    auths_per_card = 2
    data = pd.DataFrame(
        {
            "card_id": [custom_provider.card_id() for _ in range(num_cards)],
        }
    )
    data = pd.concat([data] * auths_per_card, ignore_index=True)
    data = data.assign(
        transaction_id=[
            custom_provider.transaction_id() for _ in range(num_cards * auths_per_card)
        ],
        transaction_timestamp=[
            custom_provider.transaction_timestamp()
            for _ in range(num_cards * auths_per_card)
        ],
    )
    label_name = "fraud_label"

    result = custom_provider.add_fraud_label(data, label_name=label_name)

    assert label_name in result.columns
    assert set(result[label_name].unique()).issubset({0, 1})
    assert len(result) == num_cards * auths_per_card
    # Num fraud cards
    fraud_cards = pd.Series(
        result[result[label_name] == 1]["card_id"].unique(), name="card_id"
    )
    print(result.groupby("card_id")[label_name].sum())
    assert len(fraud_cards) > 0
    assert result[label_name].sum() < len(fraud_cards) * auths_per_card


def test_add_fraud_label_5_transactions_per_card(custom_provider):
    num_cards = 10
    auths_per_card = 5
    data = pd.DataFrame(
        {
            "card_id": [custom_provider.card_id() for _ in range(num_cards)],
        }
    )
    data = pd.concat([data] * auths_per_card, ignore_index=True)
    data = data.assign(
        transaction_id=[
            custom_provider.transaction_id() for _ in range(num_cards * auths_per_card)
        ],
        transaction_timestamp=[
            custom_provider.transaction_timestamp()
            for _ in range(num_cards * auths_per_card)
        ],
    )
    label_name = "fraud_label2"

    result = custom_provider.add_fraud_label(data, label_name=label_name)

    assert label_name in result.columns
    assert set(result[label_name].unique()).issubset({0, 1})
    assert len(result) == num_cards * auths_per_card

    # Not all cards have fraud
    fraud_cards = pd.Series(
        result[result[label_name] == 1]["card_id"].unique(), name="card_id"
    )
    assert abs(len(fraud_cards) - num_cards * custom_provider.card_fraud_rate) <= 2

    # Fraud is on all later transactions from the first fraud
    fraud_start = (
        result[["card_id", "transaction_timestamp", label_name]]
        .merge(fraud_cards, on="card_id", how="inner")
        .where(lambda x: x[label_name] == 1)
        .dropna(subset=label_name)
        .groupby("card_id")
        .agg({"transaction_timestamp": "min"})
        .rename(columns={"transaction_timestamp": "actual_compromised_date"})
    )
    result = result.merge(fraud_start, on="card_id", how="left").assign(
        actual_compromised_date=lambda x: x.groupby("card_id")[
            "actual_compromised_date"
        ]
        .ffill()
        .fillna(pd.NaT),
        later_auth=lambda x: x["transaction_timestamp"] >= x["actual_compromised_date"],
        error=lambda x: (x["later_auth"] != x[label_name]).astype(int),
    )
    assert result["error"].sum() == 0

    # Not every auth on every fraud card is fraud
    non_fraud = result.merge(fraud_cards, on="card_id", how="inner").where(
        lambda x: x[label_name] == 0
    )
    assert len(non_fraud) > 0


def test_add_fraud_label_no_fraud(custom_provider):
    data = pd.DataFrame(
        {
            "card_id": [custom_provider.card_id() for _ in range(10)],
            "transaction_id": [custom_provider.transaction_id() for _ in range(10)],
            "transaction_timestamp": [
                custom_provider.transaction_timestamp() for _ in range(10)
            ],
        }
    )
    label_name = "fraud_label"

    custom_provider.card_fraud_rate = 0.0
    result = custom_provider.add_fraud_label(data, label_name=label_name)

    assert label_name in result.columns
    assert result[label_name].sum() == 0
