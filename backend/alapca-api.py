import alpaca_trade_api as tradeapi
# Set up Alpaca API
API_KEY_ID = 'PKR00SORUIRRMGBWQ5YX'
API_SECRET_KEY = '7WiIb64LsSD6KkUOKAoN4vgHRCRxswdRS6xHc8nm'
api = tradeapi.REST(API_KEY_ID, API_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

## ALPACA CODE

# Function to place a buy order
def place_buy_order(symbol, quantity):
    try:
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        print(f"Successfully placed buy order for {quantity} shares of {symbol}")
    except Exception as e:
        print(f"Error placing buy order: {e}")

# Function to place a sell order
def place_sell_order(symbol, quantity):
    try:
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        print(f"Successfully placed sell order for {quantity} shares of {symbol}")
    except Exception as e:
        print(f"Error placing sell order: {e}")


def generate_prediction_indicator(y_test_actual, future_prices_actual):
    # Calculate the difference between predicted and actual closing prices
    diff = future_prices_actual[-1] - y_test_actual[-1]

    # Determine prediction indicator based on the difference
    if diff > 0:
        return 'bullish'
    elif diff < 0:
        return 'bearish'
    else:
        return 'neutral'
