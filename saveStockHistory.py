import yfinance as yf

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JNJ", "XOM", "NVDA", "CVX",
 "META", "CSCO", "MCD", "NKE", "BMY", "DIS", "ADBE", "UPS", "CRM", "NFLX", "QCOM", "CVS", "CAT", "INTC",
 "SBUX", "AMD", "BA", "ISRG", "GE", "C", "MRNA", "MU", "GM", "UBER", "F", "VMW", "SHOP",
 "KHC", "LULU", "FDX", "BIIB", "CMG", "NEM", "SQ", "EA", "ILMN", "TEAM", "PCG",
 "GRMN", "UI", "FSLR", "PINS", "SNAP", "EXPE", "SPOT", "UTHR", "CCL", "MDB", "DBX",
 "TWLO", "DOCU", "CAR", "ROKU", "GME", "OLED", "TOL", "PTON", "LYFT", "AMC",
 "DDD", "BYND", "PRLB", "EDIT"]

for tickerIndex, tickerName in enumerate(tickers):
    print("saving " + tickerName + "...")
    ticker = yf.Ticker(tickerName)
    hist = ticker.history(period="730d", interval="1h")
    hist.to_csv('./history/' + tickerName + '_1h.csv')