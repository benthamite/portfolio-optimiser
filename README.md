# Portfolio Optimiser

This project is a portfolio optimisation tool using Modern Portfolio Theory (MPT). It allows users to input asset tickers, fetch historical data, and compute the efficient frontier.

## Features

- Input asset tickers to fetch historical data.
- Calculate expected returns, volatility, and correlation matrix.
- Display the efficient frontier.
- Download the efficient frontier data as a CSV.
- Editable start dates for historical data.
- Display and edit asset returns, volatility, and correlation matrix.

## Usage

1. Enter asset tickers separated by commas.
2. Adjust the start date for historical data.
3. View and edit asset returns, volatility, and correlation matrix.
4. View the efficient frontier plot.
5. Optionally, download the data.

## Parameters

- **Return**: The expected annual return of an asset, calculated from historical data.
- **Volatility**: The standard deviation of an asset's returns, representing risk.
- **Correlation Matrix**: A table showing the correlation coefficients between assets, indicating how they move in relation to each other.

## Screenshot

![](./Screenshot.png)

## How to run

1. Ensure you have Python and pip installed.
2. Install the required dependencies:
   ```bash
   pip install numpy pandas yfinance scipy streamlit plotly
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run portfolio_optimiser.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
