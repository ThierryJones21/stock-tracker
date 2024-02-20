export const getStockData = async (symbol, from, to)=>{
   
    const apiKey = "dc16ae239a0a90cc7f039177aa18aa33";

    const apiUrl = `https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}?apikey=${apiKey}&from=${from}&to=${to}`;

    const response = await fetch(apiUrl);

    const data = await response.json()
    return data
}