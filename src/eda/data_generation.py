import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider


class CustomProvider(BaseProvider):
    def __init__(
        self,
        generator,
        merchant_weights,
        customer_weights,
        approval_rate=0.91,
        card_fraud_rate=0.50,
    ):
        super().__init__(generator)
        self.merchant_weights = merchant_weights
        self.customer_weights = customer_weights
        self.approval_rate = approval_rate
        self.merchant_decay = 0.97
        self.card_fraud_rate = card_fraud_rate

    def transaction_type(self) -> str:
        return random.choice(
            ["CP", "CNP", "contactless", "tokenized", "ATM", "recurring"]
        )

    def merchant_name(self) -> str:
        merchants = ["Amazon", "Walmart", "Target", "Best Buy", "Costco", "Local Store"]
        return random.choices(merchants, weights=self.merchant_weights, k=1)[0]

    def customer_id(self) -> str:
        customers = [f"customer_{i}" for i in range(len(self.customer_weights))]
        return random.choices(customers, weights=self.customer_weights, k=1)[0]

    def location(self) -> str:
        return random.choice(
            ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        )

    def dollar_amount(self) -> float:
        return round(random.uniform(1.0, 1000.0), 2)

    def transaction_timestamp(self) -> str:
        return (datetime.now() - timedelta(days=random.randint(1, 365))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def card_id(self) -> str:
        return str(uuid.uuid4())

    def merchant_id(self) -> str:
        return str(uuid.uuid4())

    def transaction_id(self) -> str:
        return str(uuid.uuid4())

    def generate_decision(self) -> int:
        return 1 if random.random() < self.approval_rate else 0

    def generate_transaction(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id(),
            "card_id": self.card_id(),
            "merchant_id": self.merchant_id(),
            "merchant_name": self.merchant_name(),
            "customer_id": self.customer_id(),
            "location": self.location(),
            "transaction_type": self.transaction_type(),
            "dollar_amount": self.dollar_amount(),
            "transaction_timestamp": self.transaction_timestamp(),
            "decision": self.generate_decision(),
        }

    def add_fraud_label(
        self, data: pd.DataFrame, label_name: str = "fraud_label"
    ) -> pd.DataFrame:
        """Adds a fraud label to the given transaction data based on simulated fraud scenarios.

        This method sets a subset of cards that will have fraudulent activity based on
        the `self.card_fraud_rate`. It assigns a fraud label to transactions occurring after
        a randomly chosen start date for each compromised card.

        Args:
            data (pd.DataFrame): A DataFrame containing transaction data. It must include
                the columns 'card_id', 'transaction_id', and 'transaction_timestamp'.
            label_name (str, optional): The name of the column to store the fraud label.
                Defaults to "fraud_label".

        Returns:
            pd.DataFrame: A DataFrame with an additional column specified by `label_name`,
            indicating whether each transaction is fraudulent (1) or not (0).
        """
        # Identify cards that will have fraud on them according to self.card_fraud_rate
        card_ids = data["card_id"].unique().tolist()
        fraud_cards = pd.DataFrame(
            data=random.sample(card_ids, k=int(len(card_ids) * self.card_fraud_rate)),
            columns=["card_id"],
        )
        fraud_cards = fraud_cards.assign(fraud_on_card=1)
        data = data.merge(fraud_cards, on="card_id", how="left").assign(
            fraud_on_card=lambda x: x["fraud_on_card"].fillna(0).astype(int)
        )
        # Find a random start date for the fraud
        fraud_start = (
            data[["card_id", "transaction_id", "fraud_on_card"]]
            .where(lambda x: x.fraud_on_card == 1)
            .dropna(subset="fraud_on_card")
            .sample(frac=1.0)
            .drop_duplicates(subset=["card_id"], keep="first")
            .assign(compromised_date=1)
            .drop(columns=["fraud_on_card", "card_id"])
        )
        data = (
            data.sort_values(by=["card_id", "transaction_timestamp"])
            .merge(fraud_start, on="transaction_id", how="left")
            .assign(first_fraud=lambda x: x["compromised_date"].fillna(0).astype(int))
            .assign(
                **{
                    label_name: lambda x: x.groupby("card_id")[
                        "compromised_date"
                    ].ffill()
                }
            )
            .assign(**{label_name: lambda x: x[label_name].fillna(0).astype(int)})
        )
        return data


def generate_synthetic_transactions(num_rows: int) -> pd.DataFrame:
    # Define prior distributions for merchants and customers
    merchant_weights = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05]  # Amazon, Walmart, etc.
    customer_weights = np.random.dirichlet(np.ones(100), size=1)[0]  # 100 customers

    fake = Faker()
    provider = CustomProvider(fake, merchant_weights, customer_weights)
    fake.add_provider(provider)

    transactions = []
    for _ in range(num_rows):
        transactions.append(fake.generate_transaction())

    df = pd.DataFrame(transactions).assign(
        dollar_amount=lambda x: x["dollar_amount"].astype(float),
        transaction_timestamp=lambda x: pd.to_datetime(x["transaction_timestamp"]),
        transaction_id=lambda x: x["transaction_id"].astype(str),
        card_id=lambda x: x["card_id"].astype(str),
        merchant_id=lambda x: x["merchant_id"].astype(str),
        merchant_name=lambda x: x["merchant_name"].astype(str),
        customer_id=lambda x: x["customer_id"].astype(str),
        location=lambda x: x["location"].astype(str),
        transaction_type=lambda x: x["transaction_type"].astype(str),
    )
    df = provider.add_fraud_label(df)

    return df


if __name__ == "__main__":
    num_rows = 1000  # Specify the number of rows you want to generate
    df = generate_synthetic_transactions(num_rows)
    print(df.head())  # Display the first few rows of the generated DataFrame
    df.to_csv("synthetic_transactions.csv", index=False)  # Save to CSV
