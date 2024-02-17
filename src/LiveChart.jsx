import { symbol } from "prop-types";
import React, { useEffect, useMemo, useState } from "react";
import { getStockData } from "./service";
import { formattedStockData } from "./utils";
import ReactApexChart from "react-apexcharts";
import { candleStickOptions } from "./chart_options";

const LiveChart = ({symbol}) => {
    const [stockData, setStockdata] = useState({})

    useEffect(() =>{
        getStockData(symbol).then(data =>
            setStockdata(data)
        )
    
    }, [])
    
    const weeklySeriesData = useMemo(() => formattedStockData(stockData), [stockData])
    console.log(stockData)

    return(
        <div>
            <ReactApexChart
                series={
                    [
                        {
                            data: weeklySeriesData
                        }
                    ]
                }
                options={candleStickOptions}
                type = "candlestick"
            />
        </div>
    )
}

export default LiveChart;