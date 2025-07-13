import {
  Table,
  Button,
  Space,
  Typography,
  Layout,
  Card,
  message,
  Spin,
} from "antd";
import { useNavigate } from "react-router-dom";
import {
  CameraOutlined,
  DeleteColumnOutlined,
  LineChartOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import { useEffect, useState } from "react";
import dayjs from "dayjs";
import "./styles.css";
import axios from "axios";

const { Title } = Typography;
const { Content } = Layout;

const CellTable = () => {
  const navigate = useNavigate();
  const [dataSource, setDataSource] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCells();
  }, []);
  const fetchCells = async () => {
    try {
      const { data } = await axios.get("/api/get_cells");

      // Procesar datos
      const formatted = data?.map((item, index) => ({
        key: index + 1,
        cellId: item.cellId,
        name: item.name,
        creation_date: item.creation_date,
      }));

      setDataSource(formatted);
    } catch (err) {
      message.error("Could not load the cells table.");
      console.log(err);
    } finally {
      setLoading(false);
    }
  };
  const deleteCell = async (cellId) => {
    console.log(cellId);
    try {
      await axios.delete(`/api/delete-cell/${cellId}`);
      message.success("Cell successfully deleted");
      fetchCells(); // Refresh table
    } catch (error) {
      console.error("Error deleting cell:", error);
      message.error("Error deleting cell.");
    }
  };
  console.log(dataSource);
  const columns = [
    {
      title: "ID",
      dataIndex: "key",
      key: "key",
    },
    {
      title: "Cell ID",
      dataIndex: "cellId",
      key: "cellId",
    },
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
    },
    {
      title: "Creation Date",
      dataIndex: "creation_date",
      key: "creation_date",
      render: (date) => dayjs(date).format("DD/MM/YYYY"),
    },
    {
      title: "Actions",
      key: "actions",
      render: (_, record) => (
        <Space>
          <Button
            icon={<CameraOutlined />}
            type="primary"
            onClick={() =>
              navigate("/agregar-foto", { state: { cellId: record.cellId } })
            }
          >
            Add Photo
          </Button>
          <Button
            icon={<LineChartOutlined />}
            onClick={() =>
              navigate("/ver-grafica", { state: { cellId: record.cellId } })
            }
          >
            View Chart
          </Button>
          <Button
            icon={<DeleteColumnOutlined />}
            style={{ background: "red" }}
            onClick={() => deleteCell(record?.cellId)}
          >
            Delete
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <main className="main-layout">
      <Content className="main-content">
        <Card
          title={
            <Button
              icon={<PlusOutlined />}
              type="primary"
              style={{ width: "140px", height: "40px" }}
              onClick={() => navigate("/new-cell")}
            >
              Add New Cell
            </Button>
          }
          className="main-card"
        >
          <Title level={2} className="main-title">
            Corpus Callosum Cell Table
          </Title>

          {loading ? (
            <Spin tip="Loading cells..." />
          ) : (
            <Table
              columns={columns}
              dataSource={dataSource}
              pagination={{ pageSize: 5 }}
              rowKey="cellId"
            />
          )}
        </Card>
      </Content>
    </main>
  );
};

export default CellTable;
