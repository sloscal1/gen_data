import uuid
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml  # type: ignore


def generate_fraud_data(
    config_path: str,
    fraud_rate: float,
    fraud_noise_rate: float,
    non_fraud_noise_rate: float,
) -> pd.DataFrame:
    with open(config_path, "rt", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    gen = DataGenerator(config)
    custs = gen.generate_customers(1000, config["customer"])
    non_fraud = gen.generate_transactions(custs, gen.merchants, config["customer"])
    fraud_groups = [fraud_name for fraud_name in config if "fraud" in fraud_name]
    gen_fraud_customers = {
        fraud_group: gen.generate_fraudsters(
            customers=custs.iloc[
                i
                * len(custs)
                // len(fraud_groups) : (i + 1)
                * len(custs)
                // len(fraud_groups)
            ],
            config=config[fraud_group],
        )
        for i, fraud_group in enumerate(fraud_groups)
    }
    fraud_trxn = pd.concat(
        [
            gen.generate_transactions(fraudsters, gen.merchants, config[kind])
            for kind, fraudsters in gen_fraud_customers.items()
        ],
        axis=0,
    )
    # Randomly sample the card_ids
    good_customers = custs.sample(frac=0.8)
    fraud_customers = good_customers.sample(frac=fraud_rate)
    good_trxns = non_fraud[lambda x: x.customer_id.isin(good_customers.customer_id)]
    fraud_trxns = fraud_trxn[lambda x: x.customer_id.isin(fraud_customers.customer_id)]
    # Apply fraud_noise_rate
    fraud_trxns = fraud_trxns.assign(
        final_fraud=True,
    ).assign(
        final_fraud=lambda x: x.final_fraud
        & (np.random.rand(len(x)) >= fraud_noise_rate)
    )
    # Apply non_fraud_noise_rate
    good_trxns = good_trxns.assign(
        final_fraud=lambda x: np.random.rand(len(x)) < non_fraud_noise_rate
    )
    return pd.concat([good_trxns, fraud_trxns], axis=0)


class DataGenerator:
    def __init__(self, config: Dict[str, Any], randseed: int = 1337) -> None:
        np.random.seed(randseed)
        ## UUID does not have its random state set
        self.config = config
        self.merchants = {
            "low": self.generate_merchants(config["merchants"]["num_low"]),
            "med": self.generate_merchants(config["merchants"]["num_med"]),
            "high": self.generate_merchants(config["merchants"]["num_high"]),
        }
        self.segment_approval_rate = {
            transaction_type: rate
            for transaction_type, rate in zip(
                config["customer"]["transaction_types"],
                config["segment_approval_rate"],
            )
        }
        self.frequencies = config["frequencies"]

    def generate_customers(
        self, num_customers: int, config: Dict[str, Any]
    ) -> pd.DataFrame:
        customers = (
            pd.DataFrame(
                [uuid.uuid4() for _ in range(num_customers)], columns=["customer_id"]
            )
            .assign(
                start_date=np.random.randint(
                    0, config["start_date_offset"], size=num_customers
                ),
                end_date=lambda x: np.random.randint(
                    x.start_date + 7, config["end_date_offset"], size=num_customers
                ),
                overall=np.random.randint(
                    config["overall_trxns_low"],
                    config["overall_trxns_high"],
                    size=num_customers,
                ),
                **{
                    f"{transaction_type}_percentage": np.random.rand()
                    for transaction_type in config["transaction_types"]
                },
                **{
                    f"{transaction_type}_frequency": np.random.choice(
                        self.frequencies,
                        size=num_customers,
                        p=config[f"{transaction_type}_frequencies"],
                    )
                    for transaction_type in config["transaction_types"]
                },
                **{
                    f"{transaction_type}_time_of_day": np.random.randint(
                        config[f"{transaction_type}_time_of_day_start"] * 3600,
                        config[f"{transaction_type}_time_of_day_end"] * 3600,
                        size=num_customers,
                    )
                    for transaction_type in config["transaction_types"]
                },
                **{
                    f"{transaction_type}_amount": np.random.randint(
                        config[f"{transaction_type}_amount_low"],
                        config[f"{transaction_type}_amount_high"],
                        size=num_customers,
                    )
                    for transaction_type in config["transaction_types"]
                },
                fraud=0,
            )
            .assign(
                **{
                    f"{transaction_type}_percentage": lambda x: x[
                        f"{transaction_type}_percentage"
                    ]
                    / x[
                        [f"{ttype}_percentage" for ttype in config["transaction_types"]]
                    ].sum(axis=1)
                    for transaction_type in config["transaction_types"]
                },
                atm_amount=np.random.choice(config["atm_amounts"], size=num_customers),
            )
        )

        return customers

    def generate_fraudsters(
        self, customers: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        customers = (
            pd.DataFrame(customers["customer_id"].copy(), columns=["customer_id"])
            .assign(start_offset=np.abs(np.random.normal(5, 15, len(customers))))
            .assign(
                start_offset=lambda x: np.where(
                    x.start_offset > customers.start_date,
                    x.start_offset,
                    customers.start_date,
                ),
                start_date=lambda x: customers["end_date"] - x.start_offset,
                end_date=customers["end_date"],
                overall=np.random.randint(
                    config["overall_trxns_low"],
                    config["overall_trxns_high"],
                    size=len(customers),
                ),
                **{
                    f"{transaction_type}_percentage": np.random.rand()
                    for transaction_type in config["transaction_types"]
                },
                **{
                    f"{transaction_type}_frequency": np.random.choice(
                        self.frequencies[:5],
                        size=len(customers),
                        p=[0.01, 0.05, 0.39, 0.5, 0.05],
                    )
                    for transaction_type in config["transaction_types"]
                },
                **{
                    f"{transaction_type}_time_of_day": np.random.randint(
                        config[f"{transaction_type}_time_of_day_start"] * 3600,
                        config[f"{transaction_type}_time_of_day_end"] * 3600,
                        size=len(customers),
                    )
                    for transaction_type in config["transaction_types"]
                },
                **{
                    f"{transaction_type}_amount": np.random.randint(
                        config[f"{transaction_type}_amount_low"],
                        config[f"{transaction_type}_amount_high"],
                        size=len(customers),
                    )
                    for transaction_type in config["transaction_types"]
                },
                fraud=1,
            )
            .assign(
                **{
                    f"{transaction_type}_percentage": lambda x: x[
                        f"{transaction_type}_percentage"
                    ]
                    / x[
                        [f"{ttype}_percentage" for ttype in config["transaction_types"]]
                    ].sum(axis=1)
                    for transaction_type in config["transaction_types"]
                }
            )
            .drop(columns=["start_offset"])
        )

        return customers

    def generate_merchants(self, num_merchants: int) -> pd.DataFrame:
        merchants = pd.DataFrame(
            [uuid.uuid4() for _ in range(num_merchants)], columns=["merchant_id"]
        )
        merchants = merchants.assign(
            merchant_name="Merch_" + merchants.merchant_id.astype(str).str[:10],
            merchant_type=np.random.randint(1, 10000, num_merchants),
            zip_code=np.random.randint(0, 100000, num_merchants),
            approval_rate=1 - (np.random.rand(num_merchants) * 0.001),
        ).assign(
            zip_code=lambda x: x.zip_code.astype(str).str.zfill(5),
            merchant_type=lambda x: x.merchant_type.astype(str).str.zfill(4),
        )
        return merchants

    def generate_transactions(
        self,
        customers: pd.DataFrame,
        merchants: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        alpha = self.config["merchants"]["zipf_alpha"]
        trxns = pd.DataFrame()
        cust_cols = [
            "customer_id",
            "start_date",
            "end_date",
            "fraud",
        ]
        segment_columns = [
            "amount",
            "time_of_day",
        ]
        for transaction_type in config["transaction_types"]:
            # Get the transaction frequency and amount for each customer
            num_transactions = (
                np.ceil(
                    (
                        customers["overall"]
                        * customers[f"{transaction_type}_percentage"]
                        / (customers["end_date"] - customers["start_date"])
                        / customers[f"{transaction_type}_frequency"]
                        * 100
                    ).astype(int)
                    / 100
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
                        np.random.rand(sum(num_transactions))
                        * (x.end_date - x.start_date)
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
            low_frame: pd.DataFrame = frame[lambda x: x.amount < self.config["merchants"]["low_bound"]]  # type: ignore
            med_frame: pd.DataFrame = frame[lambda x: x.amount.between(self.config["merchants"]["low_bound"], self.config["merchants"]["high_bound"])]  # type: ignore
            high_frame: pd.DataFrame = frame[lambda x: x.amount > self.config["merchants"]["high_bound"]]  # type: ignore
            low_frame = pd.concat(
                [
                    low_frame.reset_index(drop=True),
                    merchants["low"]
                    .iloc[
                        np.clip(
                            np.random.zipf(a=alpha, size=len(low_frame)),
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
                            np.random.zipf(a=alpha, size=len(med_frame)),
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
                            np.random.zipf(a=alpha, size=len(high_frame)),
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
                < x.approval_rate * self.segment_approval_rate[transaction_type]
            )
            trxns = pd.concat(
                [trxns.reset_index(drop=True), auths.reset_index(drop=False)], axis=0
            )
        trxns = trxns.assign(card_id=lambda x: x.customer_id).drop(
            columns=["time_of_day", "amount", "approval_rate"]
        )

        return trxns
