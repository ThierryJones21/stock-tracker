import React, { useEffect, useMemo, useState } from "react";
import { getStockData } from "./service";
import { formattedStockData } from "./utils";
import ReactApexChart from "react-apexcharts";
import { candleStickOptions } from "./chart_options";

const LiveChart = () => {
  const [symbol, setSymbol] = useState('TSLA');
  const [from, setFrom] = useState("2023-10-10");
  const [to, setTo] = useState("2023-12-10");
  const [stockData, setStockData] = useState({});
  const [chartHeight, setChartHeight] = useState(600);
  const [chartWidth, setChartWidth] = useState(1000);

  const fetchData = () => {
    getStockData(symbol, from, to).then((data) => setStockData(data));
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

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleButtonClick = () => {
    fetchData();
  };

  const weeklySeriesData = useMemo(() => formattedStockData(stockData), [stockData]);

  return (
    <div>
      <div>
        <input
          type="text"
          value={symbol}
          onChange={(event) => setSymbol(event.target.value)}
          placeholder="Enter stock symbol..."
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
      <h1>{symbol}</h1>
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
