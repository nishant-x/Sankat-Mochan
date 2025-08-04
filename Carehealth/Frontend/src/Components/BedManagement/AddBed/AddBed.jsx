import React, { useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import RemoveBeds from "../RemoveBed/RemoveBeds";
import "./AddBed.css"; // Import Dark Theme CSS

const BedManagement = () => {
  const [type, setType] = useState("ICU");
  const [count, setCount] = useState(1);

  const handleAddBeds = async () => {
    try {
      const res = await axios.post("http://localhost:5000/api/beds/add-beds", { type, count });
      toast.success(res.data.message, {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
    } catch (err) {
      toast.error(err.response?.data?.message || "Error adding beds", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
    }
  };

  return (
    <div className="add-bed-container">
      {/* Toast Container */}
      <ToastContainer />

      <h1 className="add-bed-title">Manage Beds</h1>

      <div className="add-bed-form">
        <label className="add-bed-label">Type:</label>
        <select
          className="add-bed-select"
          value={type}
          onChange={(e) => setType(e.target.value)}
        >
          <option value="ICU">ICU</option>
          <option value="Private">Private</option>
          <option value="General">General</option>
        </select>

        <label className="add-bed-label">Number of Beds:</label>
        <input
          className="add-bed-input"
          type="number"
          value={count}
          onChange={(e) => setCount(parseInt(e.target.value))}
          min="1"
        />

        <button className="add-bed-button" onClick={handleAddBeds}>
          Add Beds
        </button>
      </div>

      <br />
      <p className="add-bed-remove-note">Note: Only free beds can be removed.</p>
      <RemoveBeds />
    </div>
  );
};

export default BedManagement;