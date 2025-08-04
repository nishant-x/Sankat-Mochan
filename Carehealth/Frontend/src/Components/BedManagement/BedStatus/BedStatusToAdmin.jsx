import React, { useEffect, useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./BedStatus.css";
import { useNavigate } from "react-router-dom"; // ✅ Correct import

export default function BedStatusToAdmin() {
  const [prices, setPrices] = useState({});
  const [applications, setApplications] = useState([]);
  const [bedStats, setBedStats] = useState({});
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate(); // ✅ Use useNavigate hook

  // Fetch applications and bed statistics
  const fetchData = async () => {
    try {
      const appRes = await fetch(`http://localhost:5000/api/beds/bed-status-admin`);
      const appData = await appRes.json(); // Parse JSON response

      if (appRes.ok) {
        setApplications(appData); // Assuming `data` is the array of prescriptions
      } else {
        console.error("Error from server:", appData.message || "Failed to fetch prescriptions");
        toast.error("Failed to fetch bed applications. Please try again later.", {
          position: "top-right",
          autoClose: 3000,
          hideProgressBar: false,
          closeOnClick: true,
          pauseOnHover: true,
          draggable: true,
        });
      }
      console.log("Applications:", appData);

      const statsRes = await axios.get("http://localhost:5000/api/beds/bed-stats");
      setBedStats(statsRes.data);
      console.log("Bed Stats:", statsRes.data);
    } catch (err) {
      console.error("Error fetching data:", err);
      toast.error("Error fetching bed data. Please try again later.", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
    }
  };

  // Fetch bed prices
  const fetchPrices = async () => {
    try {
      const res = await axios.get("http://localhost:5000/api/beds/get-default-prices");
      if (res.status === 200 && Array.isArray(res.data)) {
        const priceMap = res.data.reduce((acc, item) => {
          acc[item.bedType] = item.pricePerDay;
          return acc;
        }, {});
        setPrices(priceMap);
      }
    } catch (err) {
      console.error("Error fetching prices:", err);
      toast.error("Error fetching bed prices. Please try again later.", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    fetchPrices();

    // ✅ Auto-refresh every 30 seconds (optional)
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  //   const handleDischarge = async (id) => {
  //     try {
  //       // Generate the bill first
  //       const billResponse = await axios.post(`http://localhost:5000/api/bill/discharge-bill/${id}`);
  //       console.log("Discharge Bill Response:", billResponse.data);

  //       // Discharge the patient after the bill is saved
  //       const dischargeResponse = await axios.put(`http://localhost:5000/api/bill/discharge/${id}`);
  //       console.log("Discharge Response:", dischargeResponse.data);

  //       // Refresh data after successful discharge
  //       fetchData();

  //       // ✅ Corrected Navigation
  //       // navigate("/allDischargeBill", { state: billResponse.data });
  //     } catch (err) {
  //       console.error("Error discharging patient:", err);
  //     }
  //   };

  return (
    <div className="Bed-status-container">
      {/* Toast Container */}
      <ToastContainer />

      <h1 className="Bed-status-title">Bed Status</h1>

      {/* Bed Statistics */}
      <div className="Bed-status-stats">
        <h2 className="Bed-status-subtitle">Bed Statistics</h2>
        <p className="Bed-status-text">ICU Beds: {bedStats.ICU?.total} (Available: {bedStats.ICU?.available})</p>
        <p className="Bed-status-text">Private Beds: {bedStats.Private?.total} (Available: {bedStats.Private?.available})</p>
        <p className="Bed-status-text">General Beds: {bedStats.General?.total} (Available: {bedStats.General?.available})</p>
      </div>

      {/* Loading State */}
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
              {/* <th className="Bed-status-th">Action</th> */}
            </tr>
          </thead>
          <tbody>
            {applications.map((app) => {
              const allocationTime = new Date(app.allocationTime);
              const currentTime = new Date();
              const daysStayed = Math.ceil((currentTime - allocationTime) / (1000 * 60 * 60 * 24));

              // Ensure bedPrice exists before calculating totalBill
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
                  {/* <td className="Bed-status-td">
                    <button className="Bed-status-discharge-button" onClick={() => handleDischarge(app._id)}> 
                      Discharge
                    </button>
                  </td> */}
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}