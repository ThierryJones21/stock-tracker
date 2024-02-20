import React, { useEffect, useMemo, useState } from "react";
import { getStockData } from "../services/service";
import { formattedStockData } from "../services/utils";
import ReactApexChart from "react-apexcharts";
import { candleStickOptions } from "../services/constants/chart_options"
import { stockSymbols } from "../services/constants/stock_symbols"; 

const LiveChart = () => {
  const today = new Date();
  const pastDate = new Date(today);
  pastDate.setMonth(pastDate.getMonth() - 3);

  const [symbol, setSymbol] = useState("TSLA");
  const [customSymbol, setCustomSymbol] = useState("");
  const [from, setFrom] = useState(pastDate.toISOString().split("T")[0]);
  const [to, setTo] = useState(today.toISOString().split("T")[0]);
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
              <h1>{stock.name}</h1>
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
