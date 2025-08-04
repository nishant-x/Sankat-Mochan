import React from "react";
import { useNavigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./HRManagerDashboard.css";

const HRManagerDashboard = () => {
    const navigate = useNavigate();

    return (
        <div className="HR-dash-container">
            <ToastContainer />
            <h1 className="HR-dash-title">HR-Manager Dashboard</h1>
            <div className="HR-dash-grid">
                <button onClick={() => navigate("/admin-registration")} className="HR-dash-button">
                    New Admin Register
                </button>
                <button onClick={() => navigate("/Total-Doctors")} className="HR-dash-button">
                    Total Doctors
                </button>
            </div>
        </div>
    );
};

export default HRManagerDashboard;
