# Portfolio Optimiser

This project is a portfolio optimisation tool using Modern Portfolio Theory (MPT). It allows users to input asset tickers, fetch historical data, and compute the efficient frontier.

## Features

- Input asset tickers to fetch historical data.
- Calculate expected return, standard deviation, and correlation matrix.
- Display the efficient frontier with the max Sharpe ratio portfolio.
- Download the efficient frontier data as a CSV.
- Editable start dates for historical data.
- Display and edit asset returns, volatility, and correlation matrix.

## Usage

1. Enter asset tickers separated by commas.
2. Adjust the start date for historical data.
3. View and edit asset expected return, standard deviation, and correlation matrix.
4. View the efficient frontier plot.
5. Optionally, download the data as a CSV.

## Parameters

- **Expected Return**: The expected annual return of an asset.
- **Standard Deviation**: The standard deviation of an asset's returns. For example, an asset with a return of 0.10 and a standard deviation of 0.08 will yield returns between 0.02 and 0.18 (0.10 ± 0.08) in two out of three years, and between -0.04 and 0.26 (0.10 ± (0.08 × 2)) in 19 out of 20 years.
- **Correlation Matrix**: A table showing the correlation coefficients between assets, indicating how they move in relation to each other.
- **Risk-Free Rate**: The rate of return of short-term Treasury securities. Use to calculate the Sharpe ratio, a measure of risk-adjusted return (calculated as the expected return minus the risk-free rate, divided by the standard deviation).

## Screenshot

![](./Screenshot.png)

## How to run

Assuming you have git, Python and pip installed, open a terminal and follow these steps:

1. Clone this repo:
   ```bash
   git clone https://github.com/benthamite/portfolio-optimiser
   ```
2. Navigate to the project directory:
   ```bash
   cd portfolio-optimiser
   ```
3. Install the required dependencies:
   ```bash
   pip install numpy pandas yfinance scipy streamlit plotly
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run portfolio_optimiser.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
