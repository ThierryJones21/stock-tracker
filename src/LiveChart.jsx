import React, { useEffect, useMemo, useState } from "react";
import { getStockData } from "./service";
import { formattedStockData } from "./utils";
import ReactApexChart from "react-apexcharts";
import { candleStickOptions } from "./chart_options";

const LiveChart = () => {
  const [symbol, setSymbol] = useState('TSLA'); // Initialize with a default symbol
  const [stockData, setStockData] = useState({});
  const [chartHeight, setChartHeight] = useState(600);
  const [chartWidth, setChartWidth] = useState(1000);

  const fetchData = () => {
    getStockData(symbol).then((data) => setStockData(data));
    console.log(stockData)
  };

  useEffect(() => {
    fetchData(); // Fetch data on initial render
  }, [symbol]); // Re-fetch data when symbol changes

  const handleSymbolChange = (event) => {
    setSymbol(event.target.value); // Update symbol state with the entered value
  };

  const handleButtonClick = () => {
    fetchData(); // Fetch data when the button is clicked
  };

  const weeklySeriesData = useMemo(() => formattedStockData(stockData), [stockData]);

  useEffect(() => {
    const handleResize = () => {
      // Adjust height and width based on screen size
      const newHeight = window.innerHeight * 0.8; // Adjust as needed
      const newWidth = window.innerWidth * 0.8; // Adjust as needed
      setChartHeight(newHeight);
      setChartWidth(newWidth);
    };

    // Update dimensions on resize
    window.addEventListener("resize", handleResize);
    handleResize(); // Initialize dimensions

    // Cleanup on unmount
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div>
      <div>
        <input
          type="text"
          value={symbol}
          onChange={handleSymbolChange}
          placeholder="Enter stock symbol..."
        />
        <button onClick={handleButtonClick}>Update Chart</button> {/* Button to update the chart */}
      </div>
      <ReactApexChart
        series={[
          {
            data: weeklySeriesData
          }
        ]}
        options={candleStickOptions}
        type="candlestick"
        height={chartHeight}
        width={chartWidth}
      />
    </div>
  );
};

export default LiveChart;
