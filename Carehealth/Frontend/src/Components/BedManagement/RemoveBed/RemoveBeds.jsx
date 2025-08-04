import React, { useEffect, useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./RemoveBeds.css";

const RemoveBeds = () => {
  const [freeBeds, setFreeBeds] = useState([]);

  const fetchFreeBeds = async () => {
    try {
      const res = await axios.get("http://localhost:5000/api/beds/free-beds");
      setFreeBeds(res.data);
    } catch (err) {
      toast.error("Error fetching free beds");
    }
  };

  useEffect(() => {
    fetchFreeBeds();
  }, []);

  const handleRemove = async (bedNumber) => {
    try {
      await axios.delete(`http://localhost:5000/api/beds/remove-bed/${bedNumber}`);
      toast.success(`Bed ${bedNumber} removed successfully`);
      fetchFreeBeds(); // Refresh the list after removal
    } catch (err) {
      toast.error("Error removing bed");
    }
  };

  return (
    <div className="remove-bed-container">
      <ToastContainer />
      <h2 className="remove-bed-title">Remove Beds</h2>
      <p className="remove-bed-note">Note: Only free beds can be removed.</p>
      
      <div className="remove-bed-list">
        {freeBeds.length > 0 ? (
          freeBeds.map((bed) => (
            <div key={bed._id} className="remove-bed-item">
              <span className="remove-bed-info">
                {bed.bedNumber} ({bed.type})
              </span>
              <button
                className="remove-bed-button"
                onClick={() => handleRemove(bed.bedNumber)}
              >
                Remove
              </button>
            </div>
          ))
        ) : (
          <p className="remove-bed-no-beds">No free beds available for removal.</p>
        )}
      </div>
    </div>
  );
};

export default RemoveBeds;
