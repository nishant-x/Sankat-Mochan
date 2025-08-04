import React, { useEffect, useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./OccupiedBeds.css"; // Import CSS for styling

const OccupiedBeds = () => {
  const [occupiedBeds, setOccupiedBeds] = useState([]);
  const [priceUpdates, setPriceUpdates] = useState({}); // Stores price updates temporarily

  // Fetch occupied beds
  const fetchOccupiedBeds = async () => {
    try {
      const res = await axios.get("http://localhost:5000/api/beds/occupied-beds");
      setOccupiedBeds(res.data);
    } catch (err) {
      toast.error("Error fetching occupied beds");
    }
  };

  useEffect(() => {
    fetchOccupiedBeds();
  }, []);

  // Handle price update
  const handlePriceUpdate = async (bedNumber) => {
    const pricePerDay = priceUpdates[bedNumber];
    if (!pricePerDay || pricePerDay <= 0) {
      toast.error("Invalid price");
      return;
    }

    try {
      await axios.put("http://localhost:5000/api/beds/update-bed-price", { bedNumber, pricePerDay });
      toast.success(`Price for bed ${bedNumber} updated successfully`);
      setPriceUpdates({ ...priceUpdates, [bedNumber]: "" }); // Clear the input field
    } catch (err) {
      toast.error("Error updating bed price");
    }
  };

  return (
    <div className="Occupied-bed-container">
      <ToastContainer />
      <h1 className="Occupied-bed-title">Occupied Beds and Pricing</h1>

      <table className="Occupied-bed-table">
        <thead className="Occupied-bed-thead">
          <tr className="Occupied-bed-header-row">
            <th className="Occupied-bed-header">Bed Number</th>
            <th className="Occupied-bed-header">Type</th>
            <th className="Occupied-bed-header">Price Per Day (₹)</th>
            <th className="Occupied-bed-header">Action</th>
          </tr>
        </thead>
        <tbody className="Occupied-bed-tbody">
          {occupiedBeds.map((bed) => (
            <tr key={bed.bedNumber} className="Occupied-bed-row">
              <td className="Occupied-bed-cell">{bed.bedNumber}</td>
              <td className="Occupied-bed-cell">{bed.type}</td>
              <td className="Occupied-bed-cell">
                <input
                  type="number"
                  className="Occupied-bed-input"
                  value={priceUpdates[bed.bedNumber] || bed.pricePerDay}
                  onChange={(e) =>
                    setPriceUpdates({ ...priceUpdates, [bed.bedNumber]: e.target.value })
                  }
                  min="0"
                />
              </td>
              <td className="Occupied-bed-cell">
                <button
                  className="Occupied-bed-update-button"
                  onClick={() => handlePriceUpdate(bed.bedNumber)}
                >
                  Update Price
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default OccupiedBeds;
