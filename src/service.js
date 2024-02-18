// COULD ALSO TRY THIS https://www.npmjs.com/package/alphavantage
// import yahooFinance from 'yahoo-finance2';


// export const getStockData = async (symbol)=>{
//     const data = await yahooFinance.search(`${symbol}`);
   
//     return data
// }

 // const response = await fetch(`https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=${symbol}&apikey=${APIKEY}`)

const APIKEY = import.meta.env.APIKEY

export const getStockData = async (symbol, from, to)=>{
   
    const apiKey = "dc16ae239a0a90cc7f039177aa18aa33";

    const apiUrl = `https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}?apikey=${apiKey}&from=${from}&to=${to}`;

    const response = await fetch(apiUrl);

    const data = await response.json()
    return data
}