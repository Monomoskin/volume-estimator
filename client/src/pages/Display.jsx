import React, { useEffect, useState } from "react";
import axios from "axios";
import { useLocation, useParams } from "react-router-dom";
import { Spin, List, Image, Typography, message, Row, Col } from "antd";
import "./styles.css";
const { Title, Text } = Typography;

export default function CellPhotos() {
  const location = useLocation(); // obtienes el ID de la ruta
  const { cellId } = location?.state || {};
  const [loading, setLoading] = useState(false);
  const [dataSource, setDataSource] = useState([]);
  useEffect(() => {
    const fetchCellData = async () => {
      try {
        const { data } = await axios.get(`/api/get_cell/${cellId}`);
        const cellHistory = data?.photos || [];
        setDataSource(cellHistory);
      } catch (error) {
        console.error("Error al obtener datos de la célula:", error);
      } finally {
        setLoading(false);
      }
    };

    if (cellId) {
      fetchCellData();
    }
  }, [cellId]);

  if (loading) return <Spin size="large" />;
  if (!dataSource) return <div>Cell not found!</div>;

  return (
    <div style={{ padding: 20 }}>
      <div className="grid-container">
        {dataSource.map((photo, index) => (
          <div className="grid-item" key={index}>
            <Title level={4}>{photo.label}</Title>
            <Text>Created At: {photo.date}</Text>
            <br />
            <Text>Estimated Volume: {photo.volume} ml</Text>
            <br />
            <Image
              src={`/api/uploads/${photo.label}`}
              alt={photo.label}
              // className="preview-image"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
