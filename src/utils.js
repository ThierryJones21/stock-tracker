export const formattedStockData = (stock) =>{

    const formattedData = []

    if (stock['Weekly Adjusted Time Series']){
        Object.entries(
            stock['Weekly Adjusted Time Series']
        ).map(
            ([key, value]) =>{
                formattedData.push({
                    // Charting data
                    x: key,
                    y: [
                        value['1. open'],
                        value['2. high'],
                        value['3. low'],
                        value['4. close']

                    ],

                    // Other useful data for later
                    // date: key,
                    // open: parseFloat(value['1. open']),
                    // high: parseFloat(value['2. high']),
                    // low: parseFloat(value['3. low']),
                    // close: parseFloat(value['4. close']),
                    // adjustedClose: parseFloat(value['5. adjusted close']),
                    // volume: parseInt(value['6. volume']),
                    // dividendAmount: parseFloat(value['7. dividend amount'])
                    
                })
            }
        )

    }
    return formattedData;
}