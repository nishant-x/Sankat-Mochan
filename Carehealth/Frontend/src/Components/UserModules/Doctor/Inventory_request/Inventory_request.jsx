import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./Inventory_request.css";

const Inventory_request = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [selectedMed, setSelectedMed] = useState([]);
  const [selectedMedSet, setSelectedMedSet] = useState(new Set());
  const [doctor, setDoctor] = useState(null);
  const [loading, setLoading] = useState(true);

  const location = useLocation();
  const { doctor_id } = location.state || {};

  useEffect(() => {
    if (doctor_id) {
      fetch(`http://localhost:5000/doctorRoutes/doctor/${doctor_id}`)
        .then((res) => res.json())
        .then((data) => {
          if (data.doctor) {
            setDoctor(data.doctor);
          } else {
            toast.error("Doctor not found!");
          }
          setLoading(false);
        })
        .catch((err) => {
          console.error("Error fetching doctor details:", err);
          toast.error("Error fetching doctor details!");
          setLoading(false);
        });
    }
  }, [doctor_id]);

  useEffect(() => {
    if (searchTerm.trim() === "") {
      setSuggestions([]);
      return;
    }

    fetch(`http://localhost:5000/inventory/search/inventory?search=${searchTerm}`)
      .then((res) => res.json())
      .then((data) => setSuggestions(data))
      .catch((err) => console.error(err));
  }, [searchTerm]);

  const handleSelectMed = (suggestion) => {
    setSelectedMed((prevMed) => [...prevMed, { ...suggestion, quantity: 1 }]);
    setSelectedMedSet((prevSet) => new Set(prevSet).add(suggestion._id));
    setSearchTerm("");
    setSuggestions([]);
  };

  const handleRemoveMed = (id) => {
    setSelectedMed((prevMed) => prevMed.filter((med) => med._id !== id));
    setSelectedMedSet((prevSet) => {
      const updatedSet = new Set(prevSet);
      updatedSet.delete(id);
      return updatedSet;
    });
  };

  const handleFieldChange = (id, field, value) => {
    setSelectedMed((prevMed) =>
      prevMed.map((med) =>
        med._id === id ? { ...med, [field]: value } : med
      )
    );
  };

  const handleSubmit = async () => {
    try {
      const response = await fetch("http://localhost:5000/inventoryRequest/save-inventory-request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          doctorName: doctor?.name,
          department: doctor?.specialization,
          requests: selectedMed.map(({ name, quantity }) => ({ medicine_name: name, quantity })),
          grand_total: selectedMed.reduce((total, med) => total + (med.price || 0) * med.quantity, 0),
        }),
      });

      const data = await response.json();
      if (response.ok) {
        toast.success("Inventory request saved successfully!");
        setTimeout(() => navigate(-1), 2000);
      } else {
        toast.error(`Error: ${data.error}`);
      }
    } catch (err) {
      console.error("Error while saving inventory request:", err);
      toast.error("Failed to save inventory request!");
    }
  };

  if (loading) {
    return <p>Loading doctor details...</p>;
  }

  if (!doctor) {
    return <p>Doctor not found.</p>;
  }

  return (
    <div className="inventory-request-container">
      <ToastContainer position="top-right" autoClose={3000} />
      
      <div className="inventory-request-doctor-info">
        <h2>Doctor: {doctor?.name}</h2>
        <h3>Department: {doctor?.specialization}</h3>
      </div>

      <div className="inventory-request-medicine-search">
        <label>Medicine Name:</label>
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <ul>
          {suggestions.map((suggestion) =>
            !selectedMedSet.has(suggestion._id) ? (
              <li key={suggestion._id} onClick={() => handleSelectMed(suggestion)}>
                {suggestion.name}
              </li>
            ) : null
          )}
        </ul>
      </div>

      {selectedMed.length > 0 && (
        <div className="inventory-request-selected-medicines">
          <h3>Selected Medicines</h3>
          {selectedMed.map((med) => (
            <div key={med._id} className="inventory-request-medicine-item">
              <span>{med.name}</span>
              <label>
                Quantity:
                <input
                  type="number"
                  min="1"
                  value={med.quantity}
                  onChange={(e) =>
                    handleFieldChange(med._id, "quantity", e.target.value)
                  }
                />
              </label>
              <button onClick={() => handleRemoveMed(med._id)}>✖</button>
            </div>
          ))}
        </div>
      )}

      <div className="inventory-request-form-actions">
        <button onClick={handleSubmit}>Submit</button>
      </div>
    </div>
  );
};

export default Inventory_request;
