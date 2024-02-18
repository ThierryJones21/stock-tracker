import React, { useEffect, useMemo, useState } from "react";
import { getStockData } from "./service";
import { formattedStockData } from "./utils";
import ReactApexChart from "react-apexcharts";
import { candleStickOptions } from "./chart_options";
import { stockSymbols } from "./stock_symbols"; 

const LiveChart = () => {
  const [symbol, setSymbol] = useState("TSLA");
  const [customSymbol, setCustomSymbol] = useState("");
  const [from, setFrom] = useState("2023-10-10");
  const [to, setTo] = useState("2024-02-10");
  const [stockData, setStockData] = useState({});
  const [chartHeight, setChartHeight] = useState(600);
  const [chartWidth, setChartWidth] = useState(1000);

  const fetchData = () => {
    const selectedSymbol = customSymbol || symbol;
    getStockData(selectedSymbol, from, to).then((data) => setStockData(data));
  };

  useEffect(() => {
    const handleResize = () => {
      const newHeight = window.innerHeight * 0.8;
      const newWidth = window.innerWidth * 0.8;
      setChartHeight(newHeight);
      setChartWidth(newWidth);
    };

    window.addEventListener("resize", handleResize);
    handleResize();

    fetchData(); // Fetch data when component mounts

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleSymbolChange = (event) => {
    const selectedSymbol = event.target.value;
    setSymbol(selectedSymbol);
    setCustomSymbol(""); // Clear custom symbol input when dropdown selection changes
  };

  const handleButtonClick = () => {
    fetchData();
  };

  const weeklySeriesData = useMemo(() => formattedStockData(stockData), [
    stockData,
  ]);

  return (
    <div>
      <div>
        <select value={symbol} onChange={handleSymbolChange}>
          <option value="">Select a stock symbol</option>
          {stockSymbols.map((stock) => (
            <option key={stock.symbol} value={stock.symbol}>
              {stock.name}
            </option>
          ))}
        </select>
        <input
          type="text"
          value={customSymbol}
          onChange={(event) => setCustomSymbol(event.target.value)}
          placeholder="Enter custom symbol..."
        />
        <input
          type="date"
          value={from}
          onChange={(event) => setFrom(event.target.value)}
        />
        <input
          type="date"
          value={to}
          onChange={(event) => setTo(event.target.value)}
        />
        <button onClick={handleButtonClick}>Update Chart</button>
      </div>
      <h1>{customSymbol || symbol}</h1>
      <ReactApexChart
        series={[
          {
            data: weeklySeriesData,
          },
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
