import { Card, Typography, Input, DatePicker, Button, message } from "antd";
import { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const { Title } = Typography;

const NewCell = () => {
  const [name, setName] = useState("");
  const [cellId, setCellId] = useState("");
  const [date, setDate] = useState(null);
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const handleSubmit = async () => {
    if (!name || !cellId || !date) {
      message.info("Por favor, completa todos los campos.", 4);
      console.log("object");
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post("/api/add_cell", {
        id: cellId, // antes: cellId
        name: name,
        created_at: date.format("YYYY-MM-DD"), // antes: creationDate
      });

      if (response.data?.error === "Cell already exists") {
        message.warning("Ya existe una célula con ese ID.");
        return;
      }

      message.success("Célula agregada correctamente.");
      navigate("/"); // Redirige a la tabla
    } catch (error) {
      console.error(error);
      message.error("Hubo un error al agregar la célula.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-wrapper">
      <Card
        title={
          <Button
            type="primary"
            style={{ width: "120px", height: "40px" }}
            onClick={() => navigate("/")}
          >
            Back
          </Button>
        }
        className="main-card"
        style={{ maxWidth: 600, margin: "0 auto" }}
      >
        <Title level={3}>Add new cell</Title>

        <Input
          placeholder="Cell name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ marginBottom: 16 }}
        />

        <Input
          placeholder="Cell ID"
          value={cellId}
          onChange={(e) => setCellId(e.target.value)}
          style={{ marginBottom: 16 }}
        />

        <DatePicker
          placeholder="Creation date"
          value={date}
          onChange={setDate}
          format="YYYY-MM-DD"
          style={{ marginBottom: 16, width: "100%" }}
        />

        <Button
          type="primary"
          style={{ height: "40px" }}
          onClick={handleSubmit}
          block
          loading={loading}
        >
          Add Cell
        </Button>
      </Card>
    </div>
  );
};

export default NewCell;
