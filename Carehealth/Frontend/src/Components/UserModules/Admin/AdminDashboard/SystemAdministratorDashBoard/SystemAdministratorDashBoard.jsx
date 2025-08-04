import React from "react";
import { useNavigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./SystemAdministratorDashBoard.css";

const SystemAdministratorDashBoard = () => {
  const navigate = useNavigate();

  return (
    <div className="System-admin-dash-container">
      <ToastContainer />
      <h1 className="System-admin-dash-title">System Administrator Dashboard</h1>
      <div className="System-admin-dash-grid">
        <button onClick={() => navigate("/doctor-registration")} className="System-admin-dash-button-doctor">
          New Doctor Register
        </button>
        <button onClick={() => navigate("/pharmacy-registration")} className="System-admin-dash-button-pharmacy">
          New Pharmacist Register
        </button>
        <button onClick={() => navigate("/total-patients")} className="System-admin-dash-button-patients">
          Total No. of Patients
        </button>
        <button onClick={() => navigate("/bedManagement")} className="System-admin-dash-button-patients">
          Bed Management
        </button>
        <button onClick={() => navigate("/editDefaultPrices")} className="System-admin-dash-button-patients">
          Edit Bed Prize
        </button>
        <button onClick={() => navigate("/Hospital-bills")} className="System-admin-dash-button-bills">
          Hospital Pharmacy Bill Show
        </button>
      </div>
    </div>
  );
};

export default SystemAdministratorDashBoard;
