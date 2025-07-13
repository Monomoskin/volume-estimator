import React, { useEffect, useState } from "react";
import axios from "axios";
import { useLocation, useParams } from "react-router-dom";
import { Spin, List, Image, Typography, message } from "antd";

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
  if (!dataSource) return <div>No se encontró la célula</div>;

  return (
    <div style={{ padding: 20 }}>
      {/* <Title level={2}>Fotos de la célula: {cellData.name}</Title>
      <Text>Fecha de creación: {cellData.created_at}</Text> */}

      <List
        itemLayout="vertical"
        dataSource={dataSource}
        renderItem={(photo, index) => (
          <List.Item key={index}>
            <Title level={4}>{photo.label}</Title>
            <Text>Fecha: {photo.date}</Text>
            <br />
            <Text>Volumen: {photo.volume}</Text>
            <br />
            {/* Para mostrar imagen, necesitas que tu backend sirva las imágenes o una URL accesible */}
            {/* Por ejemplo si guardas la ruta o url de la imagen en el Excel, podrías hacer: */}
            <Image src={`/api/uploads/${photo.label}`} alt={photo.label} />
          </List.Item>
        )}
      />
    </div>
  );
}
