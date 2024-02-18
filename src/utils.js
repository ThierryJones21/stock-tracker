export const formattedStockData = (stock) => {
    const formattedData = [];
    console.log("in formattedStockdata func");

    if (stock['historical']) {
        stock['historical'].forEach((entry) => {
            formattedData.push({
                
                x: entry.date,
                y: [
                    entry.open,
                    entry.high,
                    entry.low,
                    entry.close
                ]
            });
        });
    }
    console.log(formattedData)
    return formattedData;
}
