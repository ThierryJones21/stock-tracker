const APIKEY = import.meta.env.ALPHAVANTAGE_KEY

export const getStockData = async (symbol)=>{
    const response = await fetch(`https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=${symbol}&apikey=${APIKEY}`)
    const data = await response.json()
    return data
}