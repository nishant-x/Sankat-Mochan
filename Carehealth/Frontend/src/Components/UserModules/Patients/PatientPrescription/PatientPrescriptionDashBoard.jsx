import React, { useState, useEffect } from 'react';
import { useLocation } from "react-router-dom";
import "./PatientPrescrption.css";

const PatientPrescrptionDashBoard = () => {
  const location = useLocation();
  const { userDetails } = location.state || {}; // Get userDetails from location.state
  const [latestPrescription, setLatestPrescription] = useState(null);

  console.log("from prescriptionsdashboard", userDetails?.patient?._id);
  console.log("patient userDetails:", userDetails);
  
  useEffect(() => {
    const fetchLatestPrescription = async () => {
      if (!userDetails?.patient?._id) {
        console.error('Patient ID is missing');
        return;
      }
  
      try {
        const response = await fetch(`http://localhost:5000/prescriptions/patient/latest-prescription/${userDetails.patient._id}`);
        const data = await response.json();
  
        if (response.ok) {
          setLatestPrescription(data);
        } else {
          console.error('Error from server:', data.message || 'Failed to fetch prescription');
        }
      } catch (err) {
        console.error('Error fetching prescription:', err);
      }
    };
  
    fetchLatestPrescription();
  }, [userDetails?.patient?._id]);
  
  if (!userDetails) {
    return <div>No user details found.</div>;
  }

  return (
    <div className="patient-prescription-container">
      <h1 className="patient-prescription-title">
        {userDetails?.name ? `Latest Prescription for ${userDetails.name}` : "Latest Prescription"}
      </h1>

      {latestPrescription ? (
        <div className="patient-prescription-card">
          <p><strong>Date:</strong> {new Date(latestPrescription.prescription_date).toLocaleDateString()}</p>

          <h3 className="patient-prescription-medicine-title">Medicines:</h3>
          {latestPrescription.prescription.length > 0 ? (
            <ul className="patient-prescription-medicine-list">
              {latestPrescription.prescription.map((med, index) => (
                <li key={index} className="patient-prescription-medicine-item">
                  <strong>Medicine:</strong> {med.medicine_name} <br />
                  <strong>Quantity:</strong> {med.quantity} <br />
                  <strong>Dosage:</strong> {med.dosage} <br />
                  <strong>Duration:</strong> {med.duration}
                </li>
              ))}
            </ul>
          ) : (
            <p className="patient-prescription-no-medicine">No medicines found in this prescription.</p>
          )}
        </div>
      ) : (
        <p className="patient-prescription-loading">Loading latest prescription...</p>
      )}
    </div>
  );
};

export default PatientPrescrptionDashBoard;
