# Exploring Modeling Impacts of Class Imbalance and Label Noise on Transaction Fraud Detection

Noisy labels exacerbate the class imbalance problem, and this is especially true in the case of fraud detection. In this project, we will explore the impact of class imbalance and label noise on transaction fraud detection using a synthetic dataset.
The dataset will be generated using a set of rules that simulate the behavior of customers and merchants. The goal is to create a dataset that is representative of real-world transaction data, while also allowing us to control for various factors including those under study: class imbalance and label noise. This setup will allow us to show how difficult it becomes to identify the best performing model, especially due to label noise.

First want to get a feel for class imbalance. As we move from balanced classes to increasingly imbalanced ones we'll see AUCPR drop

### Tasks
- [ ] Connect the data to the pertubation UI
- [ ] Change the data to generate fraud and non-fraud cards separately
- [ ] Ensure random seeds can be set to guarantee reproducibility
- [ ] Update the merchant generation to use a Zipfian distribution


* What is the relationship between card fraud rate and item fraud rate?
* Basically fraud shouldn't be found on a card more than near the end of card usage





Fraud events are generated separately. Fraud happens after a certain point and that point isn't uniform, rather it's poisson from the most recent transaction. The amounts and merchants are sampled separately from a fraud vector.


Click the button and then randomly select data until we have the correct class imbalance, generating more customers if needed or more fraud if needed.

From there, we apply label noise, with fraud events being flipped to non-fraud, and then non-fraud events being flipped to fraud uniformly at random throughout the card history.


* Generate merchant behavior
- [ ] What transaction types
- [ ] What amounts
- [ ] ATM



