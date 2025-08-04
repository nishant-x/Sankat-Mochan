import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./Add_patient_med.css";

const AddPatientMed = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [selectedMed, setSelectedMed] = useState([]);
  const [selectedMedSet, setSelectedMedSet] = useState(new Set());
  const [patientName, setPatientName] = useState("");
  const [doctorName, setDoctorName] = useState("");

  const location = useLocation();
  const { patient, doctor } = location.state || {};
  console.log(patient);

  useEffect(() => {
    if (patient && doctor) {
      setPatientName(patient.name);
      setDoctorName(doctor.name);
    }
  }, [patient, doctor]);

  useEffect(() => {
    const fetchMed = () => {
      if (searchTerm.trim() === "") {
        setSuggestions([]);
        return;
      }
      fetch(`http://localhost:5000/inventory/search/inventory?search=${searchTerm}`)
        .then((res) => res.json())
        .then((data) => setSuggestions(data))
        .catch((err) => {
          console.error(err);
          toast.error("Error fetching medicines");
        });
    };

    fetchMed();
  }, [searchTerm]);

  const handleSelectMed = (suggestion) => {
    setSelectedMed((prevMed) => [
      ...prevMed,
      { ...suggestion, quantity: 1, dose: "1x/day", duration: "1 week" },
    ]);
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
      const response = await fetch("http://localhost:5000/prescriptions/save-prescription", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patientId: patient.patientId,
          doctorId: doctor._id,
          patientName,
          doctorName,
          medicines: selectedMed,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        toast.success("Prescription saved successfully!");
        navigate(-1);
      } else {
        console.log(data.error);
        toast.error(`Error: ${data.error}`);
      }
    } catch (err) {
      console.error("Error while saving prescription:", err);
      toast.error("Error while saving prescription");
    }
  };

  return (
    <div className="component-add-patient-med-container">
      <ToastContainer />
      {/* Patient and Doctor Info */}
      <h2>Add Prescriptions</h2>
      <div className="component-add-patient-med-info">
        <label>
          Patient Name:
          <input
            type="text"
            value={patient.name}
            onChange={(e) => setPatientName(e.target.value)}
            required
          />
        </label>
        <label>
          Doctor Name:
          <input
            type="text"
            value={doctor.name}
            onChange={(e) => setDoctorName(e.target.value)}
            required
          />
        </label>
      </div>

      {/* Selected Medicines */}
      <div className="component-add-patient-med-selected">
        {selectedMed.map((med) => (
          <div className="med-item" key={med._id}>
            <h3>{med.name}</h3>
            <label>
              Quantity:
              <select
                value={med.quantity}
                onChange={(e) =>
                  handleFieldChange(med._id, "quantity", e.target.value)
                }
              >
                {[1, 2, 3, 4, 5].map((quantity) => (
                  <option key={quantity} value={quantity}>
                    {quantity}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Dosage:
              <select
                value={med.dose}
                onChange={(e) =>
                  handleFieldChange(med._id, "dose", e.target.value)
                }
              >
                {["1x/day", "2x/day", "3x/day"].map((dose) => (
                  <option key={dose} value={dose}>
                    {dose}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Duration:
              <select
                value={med.duration}
                onChange={(e) =>
                  handleFieldChange(med._id, "duration", e.target.value)
                }
              >
                {["1 week", "2 weeks", "1 month"].map((duration) => (
                  <option key={duration} value={duration}>
                    {duration}
                  </option>
                ))}
              </select>
            </label>
            <button onClick={() => handleRemoveMed(med._id)}>✖</button>
          </div>
        ))}
      </div>

      {/* Medicine Search Input */}
      <div className="component-add-patient-med-search">
        <label>Medicine Name:</label>
        <input
          type="text"
          name="medicineName"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          required
        />
        <ul>
          {suggestions.map((suggestion) => {
            return !selectedMedSet.has(suggestion._id) ? (
              <li
                key={suggestion._id}
                onClick={() => handleSelectMed(suggestion)}
              >
                {suggestion.name}
              </li>
            ) : null;
          })}
        </ul>
      </div>

      {/* Submit Button */}
      <div className="component-add-patient-med-actions">
        <button onClick={handleSubmit}>Submit</button>
      </div>
    </div>
  );
};

export default AddPatientMed;
