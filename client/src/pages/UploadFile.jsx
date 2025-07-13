import { useState } from "react";
import {
  Upload,
  Input,
  Button,
  message,
  Card,
  Image,
  DatePicker,
  Typography,
} from "antd";
import {
  UploadOutlined,
  DeleteOutlined,
  BackwardFilled,
} from "@ant-design/icons";
import axios from "axios";
import "../App.css";

import { useLocation, useNavigate } from "react-router-dom";

const { Title } = Typography;

function AddPhoto() {
  const location = useLocation();
  const [fileList, setFileList] = useState([]);
  const [zoom, setZoom] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const [date, setDate] = useState(null);
  const [volumeReady, setVolumeReady] = useState(false);
  const { cellId } = location?.state || {};
  const handleUploadChange = ({ fileList }) => setFileList(fileList);

  const handleRemove = () => {
    setZoom("");
    setResult(null);
    setVolumeReady(false);
    setFileList([]);

    setDate(null);
  };

  const handleEstimateVolume = async () => {
    if (!fileList.length || !zoom) {
      message.error("Please upload an image and specify the zoom level.");
      return;
    }

    const formData = new FormData();
    formData.append("image", fileList[0].originFileObj);
    formData.append("zoom", zoom);

    try {
      setLoading(true);

      const { data } = await axios.post("/api/predict", formData);
      setResult({ milliliters: data.volume });
      setVolumeReady(true);
      message.success(
        "Volume estimated successfully. Please complete the remaining fields."
      );
    } catch (error) {
      console.error(error);
      message.error("Error estimating volume.");
    } finally {
      setLoading(false);
    }
  };

  const handleSendCellData = async () => {
    if (!cellId || !date || !result || !fileList.length) {
      message.error("Please complete all fields before saving.");
      return;
    }

    const formData = new FormData();
    formData.append("image", fileList[0].originFileObj || fileList[0]); // file
    formData.append("zoom", zoom);
    formData.append("cell_id", cellId);
    formData.append("timestamp", date.format("YYYY-MM-DD"));
    formData.append("volume", result.milliliters);

    try {
      setLoading(true);
      await axios.post("/api/add_photo", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      message.success("Data sent successfully.");
      handleRemove(); // Reset
    } catch (error) {
      console.error(error);
      message.error("Error sending data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <Card
        title="Volume Estimation"
        bordered={false}
        className="card"
        extra={
          <Button
            icon={<BackwardFilled />}
            onClick={() => navigate("/")}
            type="primary"
          >
            Back
          </Button>
        }
      >
        {/* Image Upload */}
        {fileList.length === 0 ? (
          <Upload
            className="upload-box"
            listType="picture-card"
            fileList={fileList}
            onChange={handleUploadChange}
            beforeUpload={() => false}
          >
            <UploadOutlined />
          </Upload>
        ) : (
          <div className="preview-container">
            <Image
              className="preview-image"
              src={URL.createObjectURL(fileList[0].originFileObj)}
              alt="Preview"
            />
            <Button
              icon={<DeleteOutlined />}
              className="delete-btn"
              onClick={handleRemove}
            >
              Remove Image
            </Button>
          </div>
        )}

        {/* Zoom Input and Estimation */}
        {!volumeReady && (
          <>
            <Input
              type="number"
              placeholder="Zoom level or scale"
              value={zoom}
              onChange={(e) => setZoom(e.target.value)}
              className="input-zoom"
              suffix={"mm"}
            />

            <Button
              type="primary"
              onClick={handleEstimateVolume}
              loading={loading}
              className="submit-btn"
            >
              Estimate Volume
            </Button>
          </>
        )}

        {/* Fields after estimation */}
        {volumeReady && (
          <>
            <Title level={5}>Estimated Volume</Title>
            <Input
              value={result.milliliters}
              suffix="ml"
              disabled
              className="input-zoom"
            />

            <DatePicker
              onChange={setDate}
              placeholder="Select observation date"
              className="input-date"
              format="YYYY-MM-DD"
            />
            <Button
              type="dashed"
              onClick={handleSendCellData}
              loading={loading}
              className="submit-btn"
            >
              Submit Cell Data
            </Button>
          </>
        )}
      </Card>
    </div>
  );
}

export default AddPhoto;
