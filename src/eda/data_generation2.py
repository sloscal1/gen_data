import uuid
import pathlib

import numpy as np
import pandas as pd

transaction_types = ["cp", "cnp", "contactless", "tokenized", "atm", "recurring"]
frequencies = [0.1, 0.5, 1, 7, 30, 90, 365]
segment_approval_rate = {
    transaction_type: rate
    for transaction_type, rate in zip(
        transaction_types, [0.95, 0.9, 0.98, 0.97, 0.99, 0.96]
    )
}


def generate_customer(num_customers: int) -> pd.DataFrame:
    type_freq = np.random.rand(len(transaction_types))
    type_freq /= np.sum(type_freq)
    customers = pd.DataFrame(
        [uuid.uuid4() for _ in range(num_customers)], columns=["customer_id"]
    )
    customers = customers.assign(
        start_date=np.random.randint(0, 180, size=num_customers),
        end_date=lambda x: np.random.randint(x.start_date + 7, 365, size=num_customers),
        overall=np.random.randint(1, 1000, size=num_customers),
        cp_percentage=type_freq[0],
        cp_frequency=np.random.choice(
            frequencies, size=num_customers, p=[0.01, 0.1, 0.2, 0.3, 0.2, 0.1, 0.09]
        ),
        cp_time_of_day=np.random.randint(6 * 3600, 24 * 3600, size=num_customers),
        cp_amount=np.random.poisson(50, size=num_customers),
        cnp_percentage=type_freq[1],
        cnp_frequency=np.random.choice(
            frequencies, size=num_customers, p=[0, 0.05, 0.1, 0.2, 0.3, 0.2, 0.15]
        ),
        cnp_time_of_day=np.random.randint(0, 24 * 3600, size=num_customers),
        cnp_amount=np.random.randint(1, 2000, size=num_customers),
        contactless_percentage=type_freq[2],
        contactless_frequency=np.random.choice(
            frequencies, size=num_customers, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.05, 0.05]
        ),
        contactless_time_of_day=np.random.randint(
            6 * 3600, 24 * 3600, size=num_customers
        ),
        contactless_amount=np.random.randint(5, 2000, size=num_customers),
        tokenized_percentage=type_freq[3],
        tokenized_frequency=np.random.choice(
            frequencies, size=num_customers, p=[0.1, 0.1, 0.2, 0.25, 0.15, 0.1, 0.1]
        ),
        tokenized_time_of_day=np.random.randint(0, 24 * 3600, size=num_customers),
        tokenized_amount=np.random.randint(5, 2000, size=num_customers),
        atm_percentage=type_freq[0],
        atm_frequency=np.random.choice(
            frequencies, size=num_customers, p=[0, 0.05, 0.1, 0.2, 0.3, 0.2, 0.15]
        ),
        atm_time_of_day=np.random.randint(8 * 3600, 22 * 3600, size=num_customers),
        atm_amount=np.random.choice([20, 40, 80, 100, 200], size=num_customers),
        recurring_percentage=type_freq[5],
        recurring_frequency=np.random.choice(
            frequencies, size=num_customers, p=[0.05, 0.05, 0.05, 0.05, 0.5, 0.2, 0.1]
        ),
        recurring_time_of_day=np.random.randint(
            6 * 3600, 24 * 3600, size=num_customers
        ),
        recurring_amount=np.random.randint(5, 2000, size=num_customers),
    )
    return customers


def generate_merchants(num_merchants: int) -> pd.DataFrame:
    merchants = pd.DataFrame(
        [uuid.uuid4() for _ in range(num_merchants)], columns=["merchant_id"]
    )
    merchants = merchants.assign(
        merchant_type=np.random.randint(1, 10000, num_merchants),
        zip_code=np.random.randint(0, 100000, num_merchants),
        approval_rate=1 - (np.random.rand(num_merchants) * 0.001),
    ).assign(
        zip_code=lambda x: x.zip_code.astype(str).str.zfill(5),
        merchant_type=lambda x: x.merchant_type.astype(str).str.zfill(4),
    )
    return merchants


def generate_transactions(customers: pd.DataFrame, merchants: dict) -> pd.DataFrame:
    # Generate transactions according to the distribution parameters of each customer
    # The corresponding merchant information will be sampled from the low/med/high merchants
    # according to a zipf(a=2.2) distribution
    trxns = pd.DataFrame()
    cust_cols = [
        "customer_id",
        "start_date",
        "end_date",
    ]
    segment_columns = [
        "amount",
        "time_of_day",
    ]
    for transaction_type in transaction_types:
        # Get the transaction frequency and amount for each customer
        num_transactions = (
            (
                (customers["end_date"] - customers["start_date"])
                / customers[f"{transaction_type}_frequency"]
                * customers["overall"]
                * customers[f"{transaction_type}_percentage"]
            )
            .astype(int)
            .to_list()
        )
        # Create the customer level attribute columns
        frame = (
            pd.DataFrame(
                [uuid.uuid4() for _ in range(sum(num_transactions))],
                columns=["transaction_id"],
            )
            .assign(
                transaction_type=transaction_type,
                **{
                    col: customers[col].repeat(num_transactions).values
                    for col in cust_cols
                },
            )
            # Create the transaction level attribute columns
            .assign(
                **{
                    col: customers[f"{transaction_type}_{col}"]
                    .repeat(num_transactions)
                    .values
                    for col in segment_columns
                },
                date_offset=lambda x: pd.to_timedelta(
                    np.random.rand(sum(num_transactions)) * (x.end_date - x.start_date)
                    + x.start_date,
                    unit="D",
                ),
            )
            .assign(
                transaction_amount=lambda x: x.amount
                + np.random.normal(0, 1, sum(num_transactions)),
                transaction_timestamp=lambda x: (
                    pd.to_datetime("2024-01-01")
                    + x.date_offset
                    + pd.to_timedelta(x.time_of_day, unit="s")
                    + pd.to_timedelta(
                        np.random.normal(0, 5, sum(num_transactions)) * 60, unit="s"
                    )
                ),
            )
            .drop(
                columns=[
                    "start_date",
                    "end_date",
                    "date_offset",
                ]
            )
        )
        low_frame: pd.DataFrame = frame[lambda x: x.amount < 50]  # type: ignore
        med_frame: pd.DataFrame = frame[lambda x: x.amount.between(50, 200)]  # type: ignore
        high_frame: pd.DataFrame = frame[lambda x: x.amount > 200]  # type: ignore
        low_frame = pd.concat(
            [
                low_frame.reset_index(drop=True),
                merchants["low"]
                .iloc[
                    np.clip(
                        np.random.zipf(a=2.2, size=len(low_frame)),
                        0,
                        len(merchants["low"]) - 1,
                    )
                ]
                .reset_index(drop=True),
            ],
            axis=1,
        )
        med_frame = pd.concat(
            [
                med_frame.reset_index(drop=True),
                merchants["med"]
                .iloc[
                    np.clip(
                        np.random.zipf(a=2.2, size=len(med_frame)),
                        0,
                        len(merchants["med"]) - 1,
                    )
                ]
                .reset_index(drop=True),
            ],
            axis=1,
        )
        high_frame = pd.concat(
            [
                high_frame.reset_index(drop=True),
                merchants["high"]
                .iloc[
                    np.clip(
                        np.random.zipf(a=2.2, size=len(high_frame)),
                        0,
                        len(merchants["high"]) - 1,
                    )
                ]
                .reset_index(drop=True),
            ],
            axis=1,
        )
        auths = pd.concat([low_frame, med_frame, high_frame], axis=0).set_index(
            "transaction_id", drop=True
        )
        auths = auths.assign(
            approved=lambda x: np.random.rand(len(x))
            < x.approval_rate * segment_approval_rate[transaction_type]
        )
        trxns = pd.concat([trxns, auths], axis=0)

    return trxns

def generate_fraud_data(fraud_rate: float, fraud_noise_rate: float, non_fraud_noise_rate: float) -> pd.DataFrame:
    if not pathlib.Path("transactions.csv").exists():
        gen = DataGenerator()
        gen.transactions.to_csv("transactions.csv", index=True)
    non_fraud = pd.read_csv("transactions.csv", index_col=0, header=0)
    if not pathlib.Path("fraud.csv").exists():
        # Fraud auths get generated possion - 5 transactions from the end of the customer auths
        # We then generate uniform random n per day until m days after the last customer auth
        # Approval is an order of magnitude lower
    fraud = pd.read_csv("fraud.csv", index_col=0, header=0)
    # Randomly sample the card_ids
    # binary search to find the fraud rate
    # Add a new column with the fraud label
    # Apply fraud_noise_rate
    # Apply non_fraud_noise_rate
    return fraud


class DataGenerator:
    def __init__(self):
        self.merchants = {
            "low": generate_merchants(1000),
            "med": generate_merchants(1000),
            "high": generate_merchants(1000),
        }
        self.customers = generate_customer(10)
        self.transactions = generate_transactions(self.customers, self.merchants)


def main() -> None:

    gen = DataGenerator()
    gen.transactions.to_csv("transactions.csv", index=True)

if __name__ == "__main__":
    main()
