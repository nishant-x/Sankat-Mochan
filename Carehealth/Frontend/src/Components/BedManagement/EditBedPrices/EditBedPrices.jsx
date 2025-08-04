import React, { useState, useEffect } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./EditBedPrices.css";

const EditBedPrices = () => {
  const [prices, setPrices] = useState({ ICU: 0, Private: 0, General: 0 });
  const [loading, setLoading] = useState(false);

  // Fetch default prices from backend
  useEffect(() => {
    const fetchPrices = async () => {
      setLoading(true);
      try {
        const res = await axios.get("http://localhost:5000/api/beds/get-default-prices");
        if (res.status === 200 && Array.isArray(res.data)) {
          // Convert array to object { ICU: price, Private: price, General: price }
          const priceMap = res.data.reduce((acc, item) => {
            acc[item.bedType] = item.pricePerDay;
            return acc;
          }, {});
          setPrices(priceMap);
        }
      } catch (err) {
        toast.error("Error fetching prices. Please try again.");
      }
      setLoading(false);
    };
    fetchPrices();
  }, []);

  // Handle input change
  const handleChange = (e) => {
    const { name, value } = e.target;
    if (value < 0) return; // Prevent negative values
    setPrices((prev) => ({ ...prev, [name]: parseInt(value) || 0 }));
  };

  // Update default prices
  const handleUpdatePrices = async () => {
    setLoading(true);
    try {
      const res = await axios.put("http://localhost:5000/api/beds/update-default-prices", prices);
      if (res.status === 200) {
        toast.success("Default prices updated successfully.");
      }
    } catch (err) {
      toast.error("Error updating default prices. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="price-container">
      <ToastContainer />
      <div className="default-price-container">
        <h2 className="default-price-title">Edit Default Prices</h2>

        {Object.keys(prices).length > 0 ? (
          <>
            <label className="default-price-label" htmlFor="ICU">ICU Price:</label>
            <input 
              className="default-price-input" 
              type="number" 
              id="ICU" 
              name="ICU" 
              value={prices.ICU} 
              onChange={handleChange} 
            />

            <label className="default-price-label" htmlFor="Private">Private Room Price:</label>
            <input 
              className="default-price-input" 
              type="number" 
              id="Private" 
              name="Private" 
              value={prices.Private} 
              onChange={handleChange} 
            />

            <label className="default-price-label" htmlFor="General">General Ward Price:</label>
            <input 
              className="default-price-input" 
              type="number" 
              id="General" 
              name="General" 
              value={prices.General} 
              onChange={handleChange} 
            />

            <button 
              className="default-price-button" 
              onClick={handleUpdatePrices} 
              disabled={loading}
            >
              {loading ? "Updating..." : "Update Prices"}
            </button>
          </>
        ) : (
          <p className="default-price-loading">Loading default prices...</p>
        )}
      </div>
    </div>
  );
};

export default EditBedPrices;
