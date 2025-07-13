import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Button, Card, Typography, message, Spin } from "antd";
import ReactECharts from "echarts-for-react";
import axios from "axios";
import "./styles.css";

const { Title } = Typography;

const VerGrafica = () => {
  const location = useLocation();
  const { cellId } = location.state || {};
  const navigate = useNavigate();

  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCellData = async () => {
      try {
        const { data } = await axios.get(`/api/get_cell/${cellId}`);
        const cellHistory = data?.photos || [];

        const dates = cellHistory.map((entry) => {
          const d = new Date(entry.date);
          const day = d.getDate().toString().padStart(2, "0");
          const month = (d.getMonth() + 1).toString().padStart(2, "0");
          const year = d.getFullYear();
          return `${day}-${month}-${year}`;
        });

        const volumes = cellHistory.map((entry) => entry.volume);

        setData({ dates, volumes });
      } catch (error) {
        console.error("Error fetching cell data:", error);
        message.error("Failed to load cell data.");
      } finally {
        setLoading(false);
      }
    };

    if (cellId) {
      fetchCellData();
    }
  }, [cellId]);

  const option = {
    title: {
      text: `Estimated Volume - ${cellId}`,
      left: "center",
      textStyle: { fontSize: 20, color: "#003a8c" },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#ffffff",
      borderColor: "#91d5ff",
      borderWidth: 1,
      borderRadius: 8,
      padding: 10,
      textStyle: { color: "#000", fontSize: 14 },
      formatter: (params) => {
        const { name, value } = params[0]; // name = date, value = volume
        return `
          <div style="text-align: center; width:150px">
            <div style="font-size: 16px; color: #1890ff; display:flex; justify-content:space-between; margin-bottom:4px"><strong>📅 Date:</strong> ${name}</div>
            <div style="font-size: 16px; color: #1890ff; display:flex; justify-content:space-between;"> <strong>🧪 Volume:</strong> ${value}</div>
          </div>
        `;
      },
    },
    xAxis: {
      type: "category",
      name: "Date",
      nameLocation: "center",
      nameGap: 30,
      data: data.dates,
      axisLine: { lineStyle: { color: "#1890ff" } },
    },
    yAxis: {
      type: "value",
      name: "Volume (ml)",
      nameLocation: "center",
      nameGap: 50,
      axisLine: { lineStyle: { color: "#1890ff" } },
      splitLine: { lineStyle: { type: "dashed", color: "#e6f7ff" } },
    },
    series: [
      {
        data: data.volumes,
        type: "line",
        smooth: true,
        symbol: "circle",
        symbolSize: 8,
        itemStyle: { color: "#1890ff" },
        areaStyle: { color: "#e6f7ff" },
        lineStyle: { width: 3, color: "#1890ff" },
      },
    ],
    grid: { top: 80, left: 60, right: 40, bottom: 60 },
  };

  return (
    <div className="page-wrapper">
      <Card
        className="main-card"
        title={
          <Button
            type="primary"
            style={{ width: "120px", height: "40px" }}
            onClick={() => navigate("/")}
          >
            Back
          </Button>
        }
        extra={
          <Button
            type="primary"
            style={{ width: "120px", height: "40px" }}
            onClick={() => navigate("/cell-images", { state: { cellId } })}
          >
            View Images
          </Button>
        }
      >
        <Title level={3} className="main-title">
          Estimated Volume Chart
        </Title>
        {loading ? (
          <Spin size="large" />
        ) : (
          <ReactECharts
            option={option}
            style={{ height: 400, width: "700px" }}
          />
        )}
      </Card>
    </div>
  );
};

export default VerGrafica;
