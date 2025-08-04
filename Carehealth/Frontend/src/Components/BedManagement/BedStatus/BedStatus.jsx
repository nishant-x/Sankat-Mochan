import React, { useEffect, useState } from "react";
import axios from "axios";
import "./BedStatus.css";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";



const BedStatus = ({ doctor }) => {
  const [prices, setPrices] = useState({});
  const [applications, setApplications] = useState([]);
  const [bedStats, setBedStats] = useState({});
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const fetchData = async () => {
    try {
      console.log("Fetching", doctor._id);
      const appRes = await fetch(
        `http://localhost:5000/api/beds/bed-status-doctor/${doctor._id}`
      );
      const appData = await appRes.json();

      if (appRes.ok) {
        setApplications(appData);
        toast.success("Fetched bed applications successfully!");
      } else {
        console.error("Error from server:", appData.message || "Failed to fetch applications");
        toast.error("Failed to fetch bed applications.");
      }
      
      console.log("Applications:", appData);

      const statsRes = await axios.get("http://localhost:5000/api/beds/bed-stats");
      setBedStats(statsRes.data);
      toast.success("Fetched bed statistics successfully!");
      console.log("Bed Stats:", statsRes.data);
    } catch (err) {
      console.error("Error fetching data:", err);
      toast.error("Error fetching bed data.");
    }
  };

  const fetchPrices = async () => {
    try {
      const res = await axios.get("http://localhost:5000/api/beds/get-default-prices");
      if (res.status === 200 && Array.isArray(res.data)) {
        const priceMap = res.data.reduce((acc, item) => {
          acc[item.bedType] = item.pricePerDay;
          return acc;
        }, {});
        setPrices(priceMap);
        toast.success("Fetched bed prices successfully!");
      }
    } catch (err) {
      console.error("Error fetching prices:", err);
      toast.error("Failed to fetch bed prices.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    fetchPrices();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleDischarge = async (id) => {
    try {
      const billResponse = await axios.post(
        `http://localhost:5000/api/bill/discharge-bill/${id}`
      );
      console.log("Discharge Bill Response:", billResponse.data);
      toast.success("Bill generated successfully!");

      const dischargeResponse = await axios.put(
        `http://localhost:5000/api/bill/discharge/${id}`
      );
      console.log("Discharge Response:", dischargeResponse.data);
      toast.success("Patient discharged successfully!");

      fetchData();
    } catch (err) {
      console.error("Error discharging patient:", err);
      toast.error("Failed to discharge patient.");
    }
  };

  return (
    <div className="Bed-status-container">
      <h1 className="Bed-status-title">Bed Status</h1>

      <div className="Bed-status-stats">
        <h2 className="Bed-status-subtitle">Bed Statistics</h2>
        <p className="Bed-status-text">ICU Beds: {bedStats.ICU?.total} (Available: {bedStats.ICU?.available})</p>
        <p className="Bed-status-text">Private Beds: {bedStats.Private?.total} (Available: {bedStats.Private?.available})</p>
        <p className="Bed-status-text">General Beds: {bedStats.General?.total} (Available: {bedStats.General?.available})</p>
      </div>

      {loading ? (
        <p className="Bed-status-loading">Loading bed prices...</p>
      ) : (
        <table className="Bed-status-table">
          <thead>
            <tr className="Bed-status-table-header">
              <th className="Bed-status-th">Name</th>
              <th className="Bed-status-th">Contact</th>
              <th className="Bed-status-th">Department</th>
              <th className="Bed-status-th">Bed Type</th>
              <th className="Bed-status-th">Bed Number</th>
              <th className="Bed-status-th">Allocation Time</th>
              <th className="Bed-status-th">Days Stayed</th>
              <th className="Bed-status-th">Total Bill</th>
              <th className="Bed-status-th">Action</th>
            </tr>
          </thead>
          <tbody>
            {applications.map((app) => {
              const allocationTime = new Date(app.allocationTime);
              const currentTime = new Date();
              const daysStayed = Math.ceil((currentTime - allocationTime) / (1000 * 60 * 60 * 24));
              const bedPrice = prices[app.bedType] || 0;
              const totalBill = daysStayed * bedPrice;

              return (
                <tr key={app._id} className="Bed-status-table-row">
                  <td className="Bed-status-td">{app.name}</td>
                  <td className="Bed-status-td">{app.contact}</td>
                  <td className="Bed-status-td">{app.department}</td>
                  <td className="Bed-status-td">{app.bedType}</td>
                  <td className="Bed-status-td">{app.bedNumber}</td>
                  <td className="Bed-status-td">{allocationTime.toLocaleString()}</td>
                  <td className="Bed-status-td">{daysStayed}</td>
                  <td className="Bed-status-td">₹{totalBill}</td>
                  <td className="Bed-status-td">
                    <button
                      className="Bed-status-discharge-button"
                      onClick={() => handleDischarge(app._id)}
                    >
                      Discharge
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default BedStatus;
