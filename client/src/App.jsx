import { Routes, Route } from "react-router-dom";
import CellTable from "./pages/Table";
import AgregarFoto from "./pages/UploadFile";
import VerGrafica from "./pages/VerGrafica";
import NewCell from "./pages/NewCell";
import UploadPage from "./Upload";
import CellPhotos from "./pages/Display";

function App() {
  return (
    <Routes>
      <Route path="/" element={<CellTable />} />
      <Route path="/agregar-foto" element={<AgregarFoto />} />
      <Route path="/ver-grafica" element={<VerGrafica />} />
      <Route path="/new-cell" element={<NewCell />} />
      <Route path="/cell-images" element={<CellPhotos />} />
    </Routes>
  );
}

export default App;
