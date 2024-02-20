// import { predictStocks } from "./prediction-model";

export const formattedStockData = async ( stock ) => {
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

    //  // Call predictStocks function to get predictions asynchronously
    //  const predictions = await predictStocks(formattedData);

    //  // Add predictions to formattedData
    //  formattedData.forEach((dataPoint, index) => {
    //      dataPoint.prediction = predictions[index];
    //  });
 
     return formattedData;

}
