// COULD ALSO TRY THIS https://www.npmjs.com/package/alphavantage
// import yahooFinance from 'yahoo-finance2';


// export const getStockData = async (symbol)=>{
//     const data = await yahooFinance.search(`${symbol}`);
   
//     return data
// }

const APIKEY = import.meta.env.ALPHAVANTAGE_KEY

export const getStockData = async (symbol)=>{
    const response = await fetch(`https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=${symbol}&apikey=${APIKEY}`)
    const data = await response.json()
    return data
}