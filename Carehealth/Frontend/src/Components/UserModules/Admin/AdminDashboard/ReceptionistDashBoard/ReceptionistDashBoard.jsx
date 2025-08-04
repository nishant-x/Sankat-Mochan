import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./ReceptionistDashBoard.css";

const ReceptionistDashBoard = () => {
  const navigate = useNavigate();
  const [BedCounts, setBedCounts] = useState({ totalBeds: 0, freeBeds: 0, occupiedBeds: 0 });
  const [totalAppointmentsToday, setTotalAppointmentsToday] = useState(0);

  const fetchBedCounts = () => {
    fetch("http://localhost:5000/api/beds/totalbed")
      .then((response) => response.json())
      .then((data) => setBedCounts(data))
      .catch((error) => {
        console.error("Error fetching bed counts:", error);
        toast.error("Error fetching bed counts");
      });
  };

  const fetchTotalAppointmentsToday = () => {
    fetch("http://localhost:5000/api/appointments/total-appointment-count")
      .then((response) => response.json())
      .then((data) => {
        if (data.totalAppointments !== undefined) {
          setTotalAppointmentsToday(data.totalAppointments); // Extract the number
        } else {
          console.error("Unexpected response structure:", data);
          toast.error("Unexpected response structure for appointments");
        }
      })
      .catch((error) => {
        console.error("Error fetching total appointments:", error);
        toast.error("Error fetching total appointments");
      });
  };

  useEffect(() => {
    fetchTotalAppointmentsToday();
    fetchBedCounts();
  }, []);

  return (
    <div className="admin-dashboard-container">
      <ToastContainer />
      {/* Sidebar Section */}
      <aside className="admin-dashboard-sidebar">
        <h2 className="admin-dashboard-sidebar-title">Menu</h2>
        <div className="admin-dashboard-sidebar-buttons">
          <button className="admin-dashboard-sidebar-button" onClick={() => navigate("/")}>
            Dashboard
          </button>
          <button className="admin-dashboard-sidebar-button" onClick={() => navigate("/appointment-count")}>
            Appointments
          </button>
          <button className="admin-dashboard-sidebar-button" onClick={() => navigate("/appointment-booking")}>
            Book Appointments
          </button>
          <button className="admin-dashboard-sidebar-button" onClick={() => navigate("/Bed-status-admin")}>
            Bed Status
          </button>
          <button className="admin-dashboard-sidebar-button" onClick={() => navigate("/Hospital-bills")}>
            Inventory Bills
          </button>
        </div>
      </aside>

      {/* Main Content Section */}
      <div className="admin-dashboard-main-content-wrapper">
        {/* Header Section */}
        <header className="admin-dashboard-header">
          <h1 className="admin-dashboard-title">Hospital Management System</h1>
          <p className="admin-dashboard-welcome">Welcome, Admin</p>
        </header>

        {/* Main Content */}
        <main className="admin-dashboard-main-content">
          {/* Top Section: Quick Stats */}
          <section className="admin-dashboard-quick-stats">
            <div className="admin-dashboard-stat-card">
              <h3 className="admin-dashboard-stat-title">Total Appointments</h3>
              <p className="admin-dashboard-stat-number">{totalAppointmentsToday}</p>
              <a className="admin-dashboard-stat-link" href="/appointment-count">View Details</a>
            </div>
            <div className="admin-dashboard-stat-card">
              <h3 className="admin-dashboard-stat-title">Total Beds</h3>
              <p className="admin-dashboard-stat-number">{BedCounts.totalBeds}</p>
              <a className="admin-dashboard-stat-link" href="/bed-status">View Details</a>
            </div>
            <div className="admin-dashboard-stat-card">
              <h3 className="admin-dashboard-stat-title">Total Available Beds</h3>
              <p className="admin-dashboard-stat-number">{BedCounts.freeBeds}</p>
              <a className="admin-dashboard-stat-link" href="/bed-status">View Details</a>
            </div>
            <div className="admin-dashboard-stat-card">
              <h3 className="admin-dashboard-stat-title">Total Occupied Beds</h3>
              <p className="admin-dashboard-stat-number">{BedCounts.occupiedBeds}</p>
              <a className="admin-dashboard-stat-link" href="/bed-status">View Details</a>
            </div>
          </section>

          {/* Middle Section: Quick Actions */}
          <section className="admin-dashboard-quick-actions">
            <h2 className="admin-dashboard-actions-title">Quick Actions</h2>
            <div className="admin-dashboard-action-buttons">
              <button className="admin-dashboard-action-button" onClick={() => navigate("/editDefaultPrices")}>
                Change Bed Price
              </button>
              <button className="admin-dashboard-action-button" onClick={() => navigate("/AllDischargeBill")}>
                View Discharge Bill
              </button>
            </div>
          </section>

          {/* Bottom Section: Recent Activity */}
          <section className="admin-dashboard-recent-activity">
            <h2 className="admin-dashboard-recent-title">Recent Activity</h2>
            <div className="admin-dashboard-activity-list">
              <div className="admin-dashboard-activity-item">
                <p className="admin-dashboard-activity-text"><strong>Dr. John Doe</strong> registered a new patient.</p>
                <span className="admin-dashboard-activity-time">2 hours ago</span>
              </div>
              <div className="admin-dashboard-activity-item">
                <p className="admin-dashboard-activity-text"><strong>Bed ICUB1</strong> was allocated to a patient.</p>
                <span className="admin-dashboard-activity-time">4 hours ago</span>
              </div>
              <div className="admin-dashboard-activity-item">
                <p className="admin-dashboard-activity-text"><strong>Inventory</strong> was updated by Admin.</p>
                <span className="admin-dashboard-activity-time">6 hours ago</span>
              </div>
            </div>
          </section>
        </main>

        {/* Footer Section could be added here if needed */}
      </div>
    </div>
  );
};

export default ReceptionistDashBoard;
